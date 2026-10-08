#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四期相 Transformer 患者级五折训练

任务：
    precontrast / arterial / venous / delayed

输入：
    features_{sex}.h5、metadata_{sex}.pkl

处理流程：
1. 读取 HDF5 中已有的 acquisition group 边界和四期相标签；
2. 每组统一为 20 个图像特征：超长等距采样，不足补零并屏蔽 padding；
3. 将 1024 维图像特征投影到 256 维，通过 4 层 Transformer 分类；
4. 训练时额外构造连续子序列增强样本并加入轻微高斯噪声；
5. 损失同时考虑类别频率和原始图像数量，使用 label smoothing；
6. 按 participant_id 进行 5 折划分，男女模型分别训练；
7. 使用 AdamW、warmup + cosine 调度和验证集早停；
8. 按 macro-recall、macro-F1、validation loss 顺序选择检查点。

输出：
    fold_0~fold_4/best_phase_transformer.pth
    fold_splits.json、kfold_config.json 及各 fold 的训练记录

测试集参与者应与训练数据隔离；同一患者的多次检查不得跨 fold。
"""

import argparse
import json
import math
import os
import pickle
import random
import warnings
from collections import Counter, defaultdict
from contextlib import nullcontext

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================================
# 1. 模型与数据配置
# ============================================================================

CLASS_NAMES = ["precontrast", "arterial", "venous", "delayed"]
NUM_CLASSES = 4
METHODS_SEQUENCE_LENGTH = 20
METHODS_D_MODEL = 256
METHODS_NHEAD = 8
METHODS_NUM_LAYERS = 4
METHODS_DIM_FEEDFORWARD = 1024
METHODS_DROPOUT = 0.1
METHODS_WEIGHT_DECAY = 1e-4
METHODS_FEATURE_SCHEMA_VERSION = (
    "methods_2026_v4_sequence_boundary_0p5s_spatial256_bilinear"
)


def set_random_seed(seed: int):
    """固定 Python、NumPy 和 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 2. Transformer 模型
# ============================================================================

class PositionalEncoding(nn.Module):
    """可学习的位置编码"""

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pos_embedding(positions))


class SequenceTransformer(nn.Module):
    """
    网络结构：
        [B, 20, 1024]
            -> 1024→256 投影
            -> CLS token + 位置编码
            -> 4 层 Transformer Encoder
            -> CLS 分类头
            -> [B, 4]
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        num_classes: int = NUM_CLASSES,
        d_model: int = METHODS_D_MODEL,
        nhead: int = METHODS_NHEAD,
        num_layers: int = METHODS_NUM_LAYERS,
        dim_feedforward: int = METHODS_DIM_FEEDFORWARD,
        dropout: float = METHODS_DROPOUT,
        sequence_length: int = METHODS_SEQUENCE_LENGTH,
    ):
        super().__init__()

        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.sequence_length = int(sequence_length)

        # ConvNeXt 1024维特征投影到 256维。
        self.input_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, self.d_model) * 0.02
        )

        # +1 是因为序列前面还要添加 CLS token。
        self.pos_encoding = PositionalEncoding(
            d_model=self.d_model,
            max_len=self.sequence_length + 1,
            dropout=self.dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # 四层 Transformer。
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x.size(0)

        x = self.input_proj(x)
        x = torch.cat(
            [self.cls_token.expand(batch_size, -1, -1), x],
            dim=1,
        )

        # PyTorch Transformer 的 src_key_padding_mask：
        # False = 有效 token，True = padding，需要忽略。
        if attention_mask is not None:
            cls_mask = torch.zeros(
                batch_size,
                1,
                dtype=torch.bool,
                device=x.device,
            )
            attention_mask = torch.cat(
                [cls_mask, attention_mask],
                dim=1,
            )

        x = self.pos_encoding(x)
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=attention_mask,
        )

        return self.classifier(x[:, 0, :])


# ============================================================================
# 3. 固定 20 张序列 Dataset
# ============================================================================

class FixedLengthSequenceDataset(Dataset):
    """
    将每个 acquisition group 构造成固定 [20, feature_dim]。

    原始样本：
        >20 张：np.linspace 等间距抽取 20 张；
        =20 张：原样使用；
        <20 张：末尾补 0，并用 attention mask 屏蔽 padding。

    训练增强：
        对图像数 >3 的 acquisition group 额外生成 1 个增强样本：
        1. 随机截取原序列连续的 50%~90%；
        2. 对特征加入标准差 0.02 的轻微高斯噪声；
        3. 再按同样规则构造成固定 20 张输入。

    注意：
        - augment=False 时完全不做增强，用于验证/测试；
        - loss 中的图像数权重使用真实样本图像数，而不是 padding 后的 20。
    """

    def __init__(
        self,
        sequences: list[dict],
        sequence_length: int = METHODS_SEQUENCE_LENGTH,
        augment: bool = False,
    ):
        self.sequences = sequences
        self.sequence_length = int(sequence_length)
        self.augment = bool(augment)
        self.n_orig = len(sequences)

        if self.sequence_length != METHODS_SEQUENCE_LENGTH:
            raise ValueError(
                f"当前 Methods 要求固定 {METHODS_SEQUENCE_LENGTH} 张，"
                f"收到 sequence_length={self.sequence_length}"
            )

        if self.augment:
            self.aug_parent = [
                i for i, item in enumerate(sequences)
                if len(item["features"]) > 3
            ]
        else:
            self.aug_parent = []
        self.n_aug = len(self.aug_parent)

    def __len__(self):
        return self.n_orig + self.n_aug

    def _to_fixed_length(self, features: np.ndarray):
        """把变长特征序列转换为固定20张，并返回 padding mask。"""
        real_length = int(features.shape[0])
        feature_dim = int(features.shape[1])

        if real_length <= 0:
            raise ValueError("发现空 acquisition group")

        if real_length > self.sequence_length:
            indices = np.linspace(
                0,
                real_length - 1,
                self.sequence_length,
                dtype=np.int64,
            )
            selected = features[indices]
            valid_length = self.sequence_length
        else:
            selected = features
            valid_length = real_length

        padded = np.zeros(
            (self.sequence_length, feature_dim),
            dtype=np.float32,
        )
        padded[:valid_length] = selected[:valid_length]

        # False = 真实图像；True = zero padding，需要 Transformer 忽略。
        attention_mask = np.ones(
            self.sequence_length,
            dtype=np.bool_,
        )
        attention_mask[:valid_length] = False

        return padded, attention_mask, valid_length

    def __getitem__(self, index: int):
        if index < self.n_orig:
            item = self.sequences[index]
            features = np.asarray(item["features"], dtype=np.float32)
            label = int(item["label"])
            nimgs_for_weight = int(item.get("num_images", len(features)))
        else:
            parent_index = self.aug_parent[index - self.n_orig]
            item = self.sequences[parent_index]
            features = np.asarray(item["features"], dtype=np.float32).copy()
            label = int(item["label"])

            slen = len(features)
            min_keep = max(3, int(slen * 0.5))
            max_keep = max(min_keep, int(slen * 0.9))
            if min_keep < max_keep:
                keep_len = int(
                    torch.randint(min_keep, max_keep + 1, (1,)).item()
                )
            else:
                keep_len = min_keep

            max_start = slen - keep_len
            start = (
                int(torch.randint(0, max_start + 1, (1,)).item())
                if max_start > 0
                else 0
            )
            features = features[start:start + keep_len]

            # 轻微高斯噪声 std=0.02。
            noise = np.random.normal(
                loc=0.0,
                scale=0.02,
                size=features.shape,
            ).astype(np.float32)
            features = features + noise
            nimgs_for_weight = int(len(features))

        if features.ndim != 2:
            raise ValueError(
                f"sequence feature 应为二维 [N,D]，实际={features.shape}"
            )

        padded, attention_mask, valid_length = self._to_fixed_length(features)

        return (
            torch.from_numpy(padded),
            torch.tensor(label, dtype=torch.long),
            torch.from_numpy(attention_mask),
            torch.tensor(valid_length, dtype=torch.long),
            torch.tensor(nimgs_for_weight, dtype=torch.long),
        )


# ============================================================================
# 4. 从新版 HDF5 读取 acquisition groups
# ============================================================================

def _decode_attr(value) -> str:
    """安全读取 HDF5 字符串属性。"""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def validate_feature_metadata(metadata: dict):
    """检查特征维度、类别顺序和预处理配置是否兼容。"""
    feature_dim = int(metadata.get("feature_dim", -1))
    if feature_dim != 1024:
        raise ValueError(
            f"Methods 要求 ConvNeXt 1024维特征，metadata 中为 {feature_dim}"
        )

    class_names = list(metadata.get("class_names", []))
    if class_names != CLASS_NAMES:
        raise ValueError(
            "类别定义与当前 Methods 不一致。\n"
            f"期望: {CLASS_NAMES}\n"
            f"实际: {class_names}\n"
            "请先使用已修改的 preprocess_dicom_seq.py 和 "
            "feature_extractor_flat.py 重新生成缓存与特征。"
        )

    schema = str(metadata.get("feature_schema_version", ""))
    if schema != METHODS_FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"当前 HDF5 特征 schema={schema!r}，"
            f"当前 Methods 要求 {METHODS_FEATURE_SCHEMA_VERSION!r}。"
            "请重新运行修改后的 feature_extractor_flat.py。"
        )

    preprocessing = metadata.get("preprocessing", {})
    if preprocessing.get("interpolation") != "bilinear":
        raise ValueError(
            "特征文件未锁定为稿件要求的 bilinear 插值；"
            "请重新运行修改后的 feature_extractor_flat.py。"
        )


def load_development_sequences(
    features_dir: str,
    sex_tag: str,
) -> tuple[dict[str, list[dict]], int]:
    """
    读取 development set，并按 participant_id 聚合 acquisition groups。

    返回：
        participant_sequences:
            participant_id -> 该患者全部 examination 中的 acquisition groups
        feature_dim
    """
    h5_path = os.path.join(features_dir, f"features_{sex_tag}.h5")
    metadata_path = os.path.join(
        features_dir,
        f"metadata_{sex_tag}.pkl",
    )

    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(metadata_path)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    validate_feature_metadata(metadata)

    feature_dim = int(metadata["feature_dim"])
    examination_keys = metadata.get(
        "examination_keys",
        metadata.get("patient_ids", []),
    )

    participant_sequences = defaultdict(list)
    sequence_counter = Counter()
    examination_count = 0

    with h5py.File(h5_path, "r") as hf:
        for exam_key in tqdm(
            examination_keys,
            desc=f"读取 {sex_tag} acquisition groups",
        ):
            safe_key = str(exam_key).replace("/", "_")
            if safe_key not in hf:
                continue

            group = hf[safe_key]

            required = [
                "features",
                "sequence_ids",
                "sequence_labels",
            ]
            missing = [name for name in required if name not in group]
            if missing:
                raise ValueError(
                    f"{safe_key} 缺少新版字段 {missing}。"
                    "禁止退回按 timestamps 重新分组，以免改变 Methods。"
                )

            participant_id = _decode_attr(
                group.attrs.get("participant_id", "")
            ).strip()
            if not participant_id:
                raise ValueError(
                    f"{safe_key} 缺少 participant_id，无法保证患者级5折。"
                )

            features = group["features"][:].astype(
                np.float32,
                copy=False,
            )
            sequence_ids = group["sequence_ids"][:].astype(np.int64)
            sequence_labels = group["sequence_labels"][:].astype(np.int64)

            if len(features) != len(sequence_ids):
                raise ValueError(
                    f"{safe_key}: features 与 sequence_ids 长度不一致"
                )

            # sequence_id 已由 feature_extractor_flat.py 连续编号。
            unique_sequence_ids = []
            seen = set()
            for sid in sequence_ids.tolist():
                sid = int(sid)
                if sid not in seen:
                    seen.add(sid)
                    unique_sequence_ids.append(sid)

            for sid in unique_sequence_ids:
                if sid < 0 or sid >= len(sequence_labels):
                    raise ValueError(
                        f"{safe_key}: 非法 sequence_id={sid}"
                    )

                indices = np.flatnonzero(sequence_ids == sid)
                if len(indices) == 0:
                    continue

                label = int(sequence_labels[sid])
                if label < 0 or label >= NUM_CLASSES:
                    raise ValueError(
                        f"{safe_key}: 非法标签 {label}"
                    )

                sequence_features = features[indices]

                participant_sequences[participant_id].append({
                    "features": sequence_features,
                    "label": label,
                    "participant_id": participant_id,
                    "exam_key": str(exam_key),
                    "sequence_id": sid,
                    "num_images": int(len(indices)),
                })
                sequence_counter[label] += 1

            examination_count += 1

    if not participant_sequences:
        raise RuntimeError("没有读取到可用于训练的 acquisition groups")

    print(
        f"  {sex_tag}: {len(participant_sequences)} 个患者, "
        f"{examination_count} 次检查, "
        f"{sum(sequence_counter.values())} 个 acquisition groups"
    )
    for class_index, class_name in enumerate(CLASS_NAMES):
        print(
            f"    {class_name:<12}: "
            f"{sequence_counter.get(class_index, 0)}"
        )

    return dict(participant_sequences), feature_dim


# ============================================================================
# 5. 锁定测试集泄漏检查
# ============================================================================

def load_locked_test_participants(json_path: str | None) -> set[str]:
    """
    可选读取 locked internal test participant IDs。

    支持：
    - JSON list
    - dict 中的 locked_test_participants / test_participants / participant_ids
    """
    if not json_path:
        return set()

    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        values = obj
    elif isinstance(obj, dict):
        values = None
        for key in (
            "locked_test_participants",
            "test_participants",
            "participant_ids",
        ):
            if key in obj:
                values = obj[key]
                break
        if values is None:
            raise ValueError(
                "锁定测试集 JSON 未找到支持的 participant ID 字段"
            )
    else:
        raise ValueError("锁定测试集 JSON 必须是 list 或 dict")

    return {str(x) for x in values}


def assert_no_locked_test_leakage(
    participant_sequences: dict[str, list[dict]],
    locked_test_participants: set[str],
):
    """若 development 输入中出现 locked-test 患者，立即停止。"""
    if not locked_test_participants:
        return

    overlap = sorted(
        set(participant_sequences.keys()) & locked_test_participants
    )
    if overlap:
        examples = overlap[:20]
        raise RuntimeError(
            "检测到 locked internal test 患者出现在训练输入中，"
            "这会造成数据泄漏。\n"
            f"重叠人数={len(overlap)}，示例={examples}"
        )


# ============================================================================
# 6. Loss 与指标
# ============================================================================

def compute_class_weights(
    sequences: list[dict],
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """
    计算逆频率类别权重：
        weight_c = N / (C * N_c)
    """
    counts = Counter(int(item["label"]) for item in sequences)
    total = len(sequences)

    # 注意：四分类模型表示“全局共有4种可能类别”，
    # 并不要求每个患者、每次检查，甚至每个fold都必须同时出现4个类别。
    # 某个类别在当前fold训练集中为0是允许的。
    #
    # 类别缺失时使用安全分母，避免除零：
    #     weight_c = N / (C * max(N_c, 1))
    # 当 N_c=0 时使用 1 作为分母，避免除零，同时不改变模型的4分类输出结构。
    missing = [
        CLASS_NAMES[i]
        for i in range(NUM_CLASSES)
        if counts.get(i, 0) == 0
    ]
    if missing:
        print(
            "  ⚠ 当前fold训练集中未出现以下类别："
            f"{missing}。这是允许的；模型仍保持4分类输出。"
        )

    weights = [
        total / (NUM_CLASSES * max(counts.get(i, 0), 1))
        for i in range(NUM_CLASSES)
    ]

    tensor = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    info = {
        CLASS_NAMES[i]: {
            "count": int(counts[i]),
            "weight": float(weights[i]),
        }
        for i in range(NUM_CLASSES)
    }

    return tensor, info


def weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    nimgs: torch.Tensor,
    class_weights: torch.Tensor,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """
    两层 weighted cross-entropy：

    第一层：类别逆频率权重 class_weights；
    第二层：sequence 图像数相对权重。

    图像数权重：
        img_w = nimgs / batch_mean(nimgs)
        并限制最大为 5.0。

    """
    ce = torch.nn.functional.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        label_smoothing=label_smoothing,
        reduction="none",
    )

    image_weights = nimgs.float() / nimgs.float().mean().clamp(min=1.0)
    image_weights = image_weights.clamp(max=5.0)

    return (ce * image_weights).mean()


def compute_metrics(
    labels: list[int],
    predictions: list[int],
) -> dict:
    """计算四分类验证指标。"""
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(
            precision_score(
                labels,
                predictions,
                labels=list(range(NUM_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                labels,
                predictions,
                labels=list(range(NUM_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=list(range(NUM_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                labels=list(range(NUM_CLASSES)),
                average="weighted",
                zero_division=0,
            )
        ),
    }


# ============================================================================
# 7. 单 epoch 训练与验证
# ============================================================================

def train_epoch(
    model,
    loader,
    class_weights,
    label_smoothing: float,
    optimizer,
    device,
    scaler,
    amp_enabled: bool,
):
    model.train()
    loss_sum = 0.0
    labels_all = []
    predictions_all = []

    for features, labels, attention_mask, _, nimgs in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        nimgs = nimgs.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )

        with autocast_context:
            logits = model(
                features,
                attention_mask=attention_mask,
            )
            loss = weighted_loss(
                logits,
                labels,
                nimgs,
                class_weights,
                label_smoothing,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        predictions = logits.argmax(dim=1)

        loss_sum += float(loss.item())
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(predictions.detach().cpu().tolist())

    metrics = compute_metrics(labels_all, predictions_all)
    return loss_sum / max(len(loader), 1), metrics


@torch.no_grad()
def validate_epoch(
    model,
    loader,
    class_weights,
    label_smoothing: float,
    device,
    amp_enabled: bool,
):
    model.eval()
    loss_sum = 0.0
    labels_all = []
    predictions_all = []

    for features, labels, attention_mask, _, nimgs in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        nimgs = nimgs.to(device, non_blocking=True)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )

        with autocast_context:
            logits = model(
                features,
                attention_mask=attention_mask,
            )
            loss = weighted_loss(
                logits,
                labels,
                nimgs,
                class_weights,
                label_smoothing,
            )

        predictions = logits.argmax(dim=1)
        loss_sum += float(loss.item())
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(predictions.detach().cpu().tolist())

    metrics = compute_metrics(labels_all, predictions_all)
    cm = confusion_matrix(
        labels_all,
        predictions_all,
        labels=list(range(NUM_CLASSES)),
    )

    return (
        loss_sum / max(len(loader), 1),
        metrics,
        labels_all,
        predictions_all,
        cm,
    )


def validation_selection_key(val_metrics: dict, val_loss: float) -> tuple[float, float, float]:
    """计算模型选择排序键：优先最大化 macro-recall，其次最大化 macro-F1，
    最后最小化 validation loss。

    采用字典序比较三个指标，避免人为设置混合权重，
    同时保留 F1 和损失作为确定性 tie-breaker。
    """
    return (
        float(val_metrics["macro_recall"]),
        float(val_metrics["macro_f1"]),
        -float(val_loss),
    )


# ============================================================================
# 8. 单 fold 训练
# ============================================================================

def train_one_fold(
    fold_index: int,
    train_sequences: list[dict],
    val_sequences: list[dict],
    train_participants: list[str],
    val_participants: list[str],
    feature_dim: int,
    args,
    device,
    fold_dir: str,
):
    os.makedirs(fold_dir, exist_ok=True)

    train_dataset = FixedLengthSequenceDataset(
        train_sequences,
        augment=(not args.no_augmentation),
    )
    val_dataset = FixedLengthSequenceDataset(
        val_sequences,
        augment=False,
    )

    print(
        f"  Fold {fold_index} Dataset: 原始={train_dataset.n_orig}, "
        f"增强={train_dataset.n_aug}, 总计={len(train_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    class_weights, class_weight_info = compute_class_weights(
        train_sequences,
        device,
    )


    model = SequenceTransformer(
        feature_dim=feature_dim,
        num_classes=NUM_CLASSES,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=4,  # 模型使用四层 Encoder
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        sequence_length=METHODS_SEQUENCE_LENGTH,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 使用 warmup + cosine 学习率调度。
    warmup_epochs = min(5, max(1, args.epochs // 10))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (
            1
            + math.cos(
                math.pi
                * (epoch - warmup_epochs)
                / max(args.epochs - warmup_epochs, 1)
            )
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )

    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = (
        torch.cuda.amp.GradScaler(enabled=True)
        if amp_enabled
        else None
    )

    model_path = os.path.join(
        fold_dir,
        "best_phase_transformer.pth",
    )

    # 早停监控 validation loss；模型保存按照
    # “macro-recall 优先、macro-F1 次之、loss 作为 tie-breaker”选择。
    best_monitor_val_loss = float("inf")
    best_selection_key = None
    patience = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
        "val_accuracy": [],
    }

    print(
        f"  Fold {fold_index}: "
        f"train participants={len(train_participants)}, "
        f"val participants={len(val_participants)}, "
        f"train sequences={len(train_sequences)}, "
        f"val sequences={len(val_sequences)}"
    )
    print(f"  类别权重: {class_weight_info}")

    for epoch in range(args.epochs):
        train_loss, train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )

        (
            val_loss,
            val_metrics,
            _,
            _,
            _,
        ) = validate_epoch(
            model=model,
            loader=val_loader,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
            device=device,
            amp_enabled=amp_enabled,
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_macro_f1"].append(
            train_metrics["macro_f1"]
        )
        history["val_macro_f1"].append(
            val_metrics["macro_f1"]
        )
        history["val_accuracy"].append(
            val_metrics["accuracy"]
        )

        selection_key = validation_selection_key(val_metrics, val_loss)
        selection_improved = (
            best_selection_key is None
            or selection_key > best_selection_key
        )
        loss_improved = val_loss < (best_monitor_val_loss - args.min_delta)
        mark = " ★" if selection_improved else ""

        print(
            f"    Epoch {epoch + 1:03d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macroF1={val_metrics['macro_f1']:.4f}"
            f"{mark}",
            flush=True,
        )

        if loss_improved:
            best_monitor_val_loss = val_loss
            patience = 0
        else:
            patience += 1

        if selection_improved:
            best_selection_key = selection_key
            torch.save(
                {
                    "epoch": epoch,
                    "fold": fold_index,
                    "model_state_dict": model.state_dict(),
                    "model_type": "SequenceTransformer",
                    "task": "phase_4class",
                    "feature_dim": feature_dim,
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "num_layers": 4,
                    "dim_feedforward": args.dim_feedforward,
                    "dropout": args.dropout,
                    "sequence_length": METHODS_SEQUENCE_LENGTH,
                    "max_seq_len": METHODS_SEQUENCE_LENGTH,
                    "sampling_strategy_over_20": "uniform_np_linspace",
                    "padding_strategy_under_20": "zero_padding_with_attention_mask",
                    "class_weight_formula": "N / (C * N_c)",
                    "class_weights": class_weight_info,
                    "loss": "weighted_cross_entropy",
                    "label_smoothing": float(args.label_smoothing),
                    "image_count_weighting": {
                        "enabled": True,
                        "formula": "nimgs / batch_mean(nimgs)",
                        "max_weight": 5.0,
                    },
                    "augmentation": {
                        "enabled": bool(not args.no_augmentation),
                        "strategy": "contiguous_subsequence_50_90_percent_plus_gaussian_noise",
                        "gaussian_noise_std": 0.02,
                    },
                    "early_stopping_metric": "validation_loss",
                    "best_val_loss": float(val_loss),
                    "best_monitor_val_loss": float(best_monitor_val_loss),
                    "selection_metric": (
                        "macro_recall_then_macro_f1_then_validation_loss"
                    ),
                    "selection_key": [float(x) for x in selection_key],
                    "val_metrics": val_metrics,
                    "train_participants": train_participants,
                    "val_participants": val_participants,
                    "args": vars(args),
                },
                model_path,
            )

        if patience >= args.early_stop_patience:
            print(
                f"    早停：连续 {patience} 个 epoch "
                "validation loss 未改善"
            )
            break

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Fold {fold_index} 未保存任何模型: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    (
        final_val_loss,
        final_metrics,
        final_labels,
        final_predictions,
        final_cm,
    ) = validate_epoch(
        model=model,
        loader=val_loader,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        device=device,
        amp_enabled=amp_enabled,
    )

    # 保存训练曲线。
    fig = plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted cross-entropy")
    plt.title(f"Fold {fold_index} Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(
        os.path.join(fold_dir, "training_loss.png"),
        dpi=150,
    )
    plt.close(fig)

    result = {
        "fold": fold_index,
        "best_epoch": int(checkpoint["epoch"] + 1),
        "val_loss": float(final_val_loss),
        **final_metrics,
        "confusion_matrix": final_cm.tolist(),
        "train_participants": train_participants,
        "val_participants": val_participants,
        "model_path": model_path,
    }

    with open(
        os.path.join(fold_dir, "fold_result.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"  Fold {fold_index} 最优 epoch={result['best_epoch']}, "
        f"val_loss={result['val_loss']:.4f}, "
        f"acc={result['accuracy']:.4f}, "
        f"macroF1={result['macro_f1']:.4f}"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================================
# 9. 5折患者级交叉验证
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "按当前 Methods 训练四期相 participant-level 5-fold Transformer"
        )
    )

    parser.add_argument(
        "--features_dir",
        required=True,
        help="development set 的特征目录；不得包含 locked internal test set",
    )
    parser.add_argument(
        "--sex_tag",
        required=True,
        choices=["male", "female"],
        help="男女分别训练独立模型",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
    )
    parser.add_argument(
        "--locked_test_participants_json",
        default=None,
        help=(
            "可选：锁定内部测试集 participant ID 列表；"
            "若与 development 输入有任何重叠则立即报错"
        ),
    )

    # 固定使用五折交叉验证，不开放折数参数。
    parser.add_argument("--d_model", type=int, default=METHODS_D_MODEL)
    parser.add_argument("--nhead", type=int, default=METHODS_NHEAD)
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=METHODS_DIM_FEEDFORWARD,
        help="稿件锁定的 Transformer FFN hidden dimension",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=METHODS_DROPOUT,
        help="稿件锁定的 Transformer dropout",
    )

    # 训练超参数的默认配置。
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=METHODS_WEIGHT_DECAY,
        help="稿件锁定的 AdamW weight decay",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.05,
        help="默认0.05",
    )
    parser.add_argument(
        "--no_augmentation",
        action="store_true",
        help="关闭子序列增强；默认保留增强",
    )
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="显式开启 CUDA AMP；默认关闭，以便复现时参数更透明",
    )

    args = parser.parse_args()

    set_random_seed(args.seed)

    if args.output_dir is None:
        args.output_dir = (
            f"./transformer_phase_{args.sex_tag}_kfold_methods"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        f"cuda:{args.device_id}"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 76)
    print("四期相 Transformer | 当前 Methods 版本")
    print(f"sex={args.sex_tag} | device={device}")
    print(
        "task=precontrast/arterial/venous/delayed | "
        "participant-level 5-fold"
    )
    print(
        f"sequence_length={METHODS_SEQUENCE_LENGTH} | "
        "1024->256 | Transformer layers=4"
    )
    print("locked internal test set 不参与训练和模型选择")
    print("=" * 76)

    participant_sequences, feature_dim = load_development_sequences(
        features_dir=args.features_dir,
        sex_tag=args.sex_tag,
    )

    locked_test_participants = load_locked_test_participants(
        args.locked_test_participants_json
    )
    assert_no_locked_test_leakage(
        participant_sequences,
        locked_test_participants,
    )

    participant_ids = np.asarray(
        sorted(participant_sequences.keys()),
        dtype=object,
    )

    if len(participant_ids) < 5:
        raise ValueError(
            f"患者数={len(participant_ids)}，不足以进行5折交叉验证"
        )

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=args.seed,
    )

    fold_splits = []
    fold_results = []

    for fold_index, (train_index, val_index) in enumerate(
        kfold.split(participant_ids)
    ):
        train_participants = participant_ids[train_index].tolist()
        val_participants = participant_ids[val_index].tolist()

        # 显式验证患者不跨 fold。
        overlap = set(train_participants) & set(val_participants)
        if overlap:
            raise RuntimeError(
                f"Fold {fold_index} 出现患者泄漏: {sorted(overlap)[:10]}"
            )

        train_sequences = [
            sequence
            for participant_id in train_participants
            for sequence in participant_sequences[participant_id]
        ]
        val_sequences = [
            sequence
            for participant_id in val_participants
            for sequence in participant_sequences[participant_id]
        ]

        train_counts = Counter(
            item["label"] for item in train_sequences
        )
        val_counts = Counter(
            item["label"] for item in val_sequences
        )

        print("\n" + "-" * 76)
        print(f"FOLD {fold_index}/4")
        print(
            f"  train participants={len(train_participants)} | "
            f"val participants={len(val_participants)}"
        )
        print(
            "  train classes="
            + str({
                CLASS_NAMES[i]: train_counts.get(i, 0)
                for i in range(NUM_CLASSES)
            })
        )
        print(
            "  val classes="
            + str({
                CLASS_NAMES[i]: val_counts.get(i, 0)
                for i in range(NUM_CLASSES)
            })
        )

        split_info = {
            "fold": fold_index,
            "train_participants": train_participants,
            "val_participants": val_participants,
            "train_sequence_count": len(train_sequences),
            "val_sequence_count": len(val_sequences),
            "train_class_counts": {
                CLASS_NAMES[i]: int(train_counts.get(i, 0))
                for i in range(NUM_CLASSES)
            },
            "val_class_counts": {
                CLASS_NAMES[i]: int(val_counts.get(i, 0))
                for i in range(NUM_CLASSES)
            },
        }
        fold_splits.append(split_info)

        fold_dir = os.path.join(
            args.output_dir,
            f"fold_{fold_index}",
        )

        result = train_one_fold(
            fold_index=fold_index,
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            train_participants=train_participants,
            val_participants=val_participants,
            feature_dim=feature_dim,
            args=args,
            device=device,
            fold_dir=fold_dir,
        )
        fold_results.append(result)

    with open(
        os.path.join(args.output_dir, "fold_splits.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            fold_splits,
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary = {
        "methods_version": "2026_methods_4class_fixed20",
        "task": "phase_4class",
        "sex_tag": args.sex_tag,
        "n_folds": 5,
        "class_names": CLASS_NAMES,
        "feature_dim": feature_dim,
        "sequence_length": METHODS_SEQUENCE_LENGTH,
        "sampling_strategy_over_20": "uniform_np_linspace",
        "padding_strategy_under_20": "zero_padding_with_attention_mask",
        "model": {
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": 4,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
        },
        "optimizer": "AdamW",
        "weight_decay": float(args.weight_decay),
        "loss": "weighted_cross_entropy",
        "class_weight_formula": "N / (C * N_c)",
        "early_stopping": {
            "monitor": "validation_loss",
            "patience": args.early_stop_patience,
            "min_delta": args.min_delta,
        },
        "model_selection": (
            "macro_recall_then_macro_f1_then_validation_loss"
        ),
        "participant_disjoint_cv": True,
        "locked_test_used_in_training": False,
        "fold_results": fold_results,
        "mean_accuracy": float(
            np.mean([x["accuracy"] for x in fold_results])
        ),
        "mean_macro_f1": float(
            np.mean([x["macro_f1"] for x in fold_results])
        ),
        "args": vars(args),
    }

    with open(
        os.path.join(args.output_dir, "kfold_config.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 76)
    print("5折训练完成")
    for result in fold_results:
        print(
            f"  Fold {result['fold']}: "
            f"acc={result['accuracy']:.4f}, "
            f"macroF1={result['macro_f1']:.4f}, "
            f"best_epoch={result['best_epoch']}"
        )
    print(
        f"  Mean accuracy={summary['mean_accuracy']:.4f}, "
        f"Mean macroF1={summary['mean_macro_f1']:.4f}"
    )
    print(
        "  模型路径: "
        f"{args.output_dir}/fold_*/best_phase_transformer.pth"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
