#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四期相 Transformer 五折集成推理与结果评估

输入：
    features_{sex}.h5、metadata_{sex}.pkl
    fold_0~fold_4/best_phase_transformer.pth

流程：
1. 读取 HDF5 中的 acquisition group，不重新按时间戳分组；
2. 每组统一为 20 个图像特征，生成与训练一致的 padding mask；
3. 五个 fold 分别计算四类概率，求平均后取 argmax 得到 Category_Raw；
4. 按配置生成 Category_Predict，并记录是否应用顺序修正；
5. 输出 acquisition-group 级分类指标和 examination 级准确率；
6. 按 participant 聚类 Bootstrap 计算区间，并计算 delayed 精确区间；
7. 支持单性别评估或通过 PatientSex 将男女样本路由至对应模型。

后处理说明：
    默认在缺少已确认的顺序修正规则时保留 Raw 作为占位输出；
    reconstructed_ordering 为可选的顺序约束实现，两种模式在结果中分别标记。

同一 participant 可包含多次 examination；一次检查可以只包含部分期相。
"""

import argparse
import csv
import glob
import json
import os
import pickle
import random
import warnings
from collections import Counter, defaultdict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

try:
    from scipy.stats import beta as beta_distribution
except Exception:
    beta_distribution = None

warnings.filterwarnings("ignore")


# ============================================================================
# 1. 固定配置
# ============================================================================

CLASS_NAMES = ["precontrast", "arterial", "venous", "delayed"]
NUM_CLASSES = 4
SEQUENCE_LENGTH = 20
METHODS_D_MODEL = 256
METHODS_NHEAD = 8
METHODS_NUM_LAYERS = 4
METHODS_DIM_FEEDFORWARD = 1024
METHODS_DROPOUT = 0.1
METHODS_FEATURE_SCHEMA_VERSION = (
    "methods_2026_v4_sequence_boundary_0p5s_spatial256_bilinear"
)
PHASE_ORDER = {name: index for index, name in enumerate(CLASS_NAMES)}

BOOTSTRAP_ITERATIONS_DEFAULT = 10000
BOOTSTRAP_SEED_DEFAULT = 20260922


# ============================================================================
# 2. Transformer 模型定义
# ============================================================================

class PositionalEncoding(nn.Module):
    """与训练脚本一致的可学习位置编码。"""

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pos_embedding(positions))


class SequenceTransformer(nn.Module):
    """
        [B, 20, 1024]
            -> 1024 -> 256
            -> CLS token + 位置编码
            -> 4 层 Transformer Encoder
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
        sequence_length: int = SEQUENCE_LENGTH,
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

        self.input_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, self.d_model) * 0.02
        )

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

        # False = 有效 token；True = padding，需要 Transformer 忽略。
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
# 3. 模型加载
# ============================================================================

def _checkpoint_sequence_length(checkpoint: dict) -> int:
    """兼容之前的训练脚本中的 sequence_length / max_seq_len 字段。"""
    if "sequence_length" in checkpoint:
        return int(checkpoint["sequence_length"])
    if "max_seq_len" in checkpoint:
        return int(checkpoint["max_seq_len"])
    return SEQUENCE_LENGTH


def load_single_phase_model(path: str, device: torch.device):
    """加载一个四分类 fold 模型，并严格核对结构。"""
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if checkpoint.get("model_type", "SequenceTransformer") != "SequenceTransformer":
        raise ValueError(f"{path} 不是 SequenceTransformer checkpoint")

    num_classes = int(checkpoint.get("num_classes", -1))
    class_names = list(checkpoint.get("class_names", []))
    sequence_length = _checkpoint_sequence_length(checkpoint)

    if num_classes != NUM_CLASSES:
        raise ValueError(
            f"{path} 的 num_classes={num_classes}，当前四分类评估要求 {NUM_CLASSES}。\n"
        )

    if class_names and class_names != CLASS_NAMES:
        raise ValueError(
            f"{path} 的 class_names={class_names}，与当前定义 {CLASS_NAMES} 不一致"
        )

    if sequence_length != SEQUENCE_LENGTH:
        raise ValueError(
            f"{path} 的 sequence_length={sequence_length}，当前评估固定为 {SEQUENCE_LENGTH}"
        )

    expected_architecture = {
        "d_model": METHODS_D_MODEL,
        "nhead": METHODS_NHEAD,
        "num_layers": METHODS_NUM_LAYERS,
        "dim_feedforward": METHODS_DIM_FEEDFORWARD,
        "dropout": METHODS_DROPOUT,
    }
    for key, expected in expected_architecture.items():
        actual = checkpoint.get(key, expected)
        if key == "dropout":
            matches = np.isclose(float(actual), float(expected))
        else:
            matches = int(actual) == int(expected)
        if not matches:
            raise ValueError(
                f"{path} 的 {key}={actual} 与投稿稿件锁定值 {expected} 不一致；"
                "请使用当前 train_transformer_phase_kfold.py 重新训练，"
                "禁止用旧架构权重生成投稿结果。"
            )

    expected_selection = "macro_recall_then_macro_f1_then_validation_loss"
    selection_metric = checkpoint.get("selection_metric")
    if selection_metric != expected_selection:
        raise ValueError(
            f"{path} 的 selection_metric={selection_metric!r} 与稿件锁定规则不一致；"
            "请使用当前训练脚本重新训练并保存模型。"
        )

    model = SequenceTransformer(
        feature_dim=int(checkpoint.get("feature_dim", 1024)),
        num_classes=num_classes,
        d_model=int(checkpoint.get("d_model", METHODS_D_MODEL)),
        nhead=int(checkpoint.get("nhead", METHODS_NHEAD)),
        num_layers=int(checkpoint.get("num_layers", METHODS_NUM_LAYERS)),
        dim_feedforward=int(
            checkpoint.get("dim_feedforward", METHODS_DIM_FEEDFORWARD)
        ),
        dropout=float(checkpoint.get("dropout", METHODS_DROPOUT)),
        sequence_length=sequence_length,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    return model, checkpoint


def load_kfold_models(model_dir: str, device: torch.device):
    """从 fold_0 ~ fold_4 加载 5 个四分类模型。"""
    fold_dirs = sorted(glob.glob(os.path.join(model_dir, "fold_*")))
    if not fold_dirs:
        raise FileNotFoundError(f"未找到 fold_* 目录: {model_dir}")

    models = []
    checkpoints = []

    for fold_dir in fold_dirs:
        preferred = os.path.join(fold_dir, "best_phase_transformer.pth")
        if os.path.exists(preferred):
            model_path = preferred
        else:
            candidates = sorted(
                glob.glob(os.path.join(fold_dir, "best_*_transformer.pth"))
            )
            if not candidates:
                continue
            model_path = candidates[0]

        model, checkpoint = load_single_phase_model(model_path, device)
        models.append(model)
        checkpoints.append(checkpoint)

        fold_index = checkpoint.get("fold", len(models) - 1)
        best_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_loss = checkpoint.get("best_val_loss", None)
        if best_val_loss is None:
            extra = ""
        else:
            extra = f", val_loss={float(best_val_loss):.6f}"
        print(
            f"  Fold {fold_index}: epoch={best_epoch}{extra} "
            f"<- {model_path}"
        )

    if len(models) != 5:
        raise RuntimeError(
            f"Methods 要求 5-fold probability ensemble，但实际只加载到 {len(models)} 个模型"
        )

    # 再次核对 5 个 fold 的核心结构完全一致。
    reference = checkpoints[0]
    keys_to_check = [
        "feature_dim",
        "num_classes",
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "dropout",
    ]
    for index, checkpoint in enumerate(checkpoints[1:], start=1):
        for key in keys_to_check:
            if checkpoint.get(key) != reference.get(key):
                raise ValueError(
                    f"fold_{index} 的 {key}={checkpoint.get(key)} 与 fold_0={reference.get(key)} 不一致"
                )

    return models, checkpoints


# ============================================================================
# 4. 特征读取与固定 20 张输入
# ============================================================================

def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def validate_feature_metadata(metadata: dict):
    """确认特征文件来自当前 sequence-boundary 版本。"""
    if int(metadata.get("feature_dim", -1)) != 1024:
        raise ValueError(
            f"feature_dim={metadata.get('feature_dim')}，当前 Methods 要求 1024"
        )

    class_names = list(metadata.get("class_names", []))
    if class_names != CLASS_NAMES:
        raise ValueError(
            f"特征类别定义={class_names}，当前要求={CLASS_NAMES}"
        )

    schema = str(metadata.get("feature_schema_version", ""))
    if schema != METHODS_FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"当前特征 schema={schema!r}，当前 Methods 要求 "
            f"{METHODS_FEATURE_SCHEMA_VERSION!r}；请先使用修改后的 "
            "feature_extractor_flat.py 重新生成。"
        )

    preprocessing = metadata.get("preprocessing", {})
    if preprocessing.get("interpolation") != "bilinear":
        raise ValueError(
            "特征文件未锁定为稿件要求的 bilinear 插值；"
            "请重新运行修改后的 feature_extractor_flat.py。"
        )


def load_allowed_participants(json_path: str | None) -> set[str] | None:
    """
    读取测试 participant ID。

    支持：
      - JSON list
      - dict 中的 locked_test_participants / test_participants / participant_ids
      - dict 中 male / female 子列表（调用方会提前选择对应性别）
    """
    if not json_path:
        return None

    with open(json_path, "r", encoding="utf-8") as file:
        obj = json.load(file)

    if isinstance(obj, list):
        return {str(x) for x in obj}

    if not isinstance(obj, dict):
        raise ValueError("test_participants_json 必须是 list 或 dict")

    for key in (
        "locked_test_participants",
        "test_participants",
        "participant_ids",
    ):
        if key in obj:
            return {str(x) for x in obj[key]}

    raise ValueError(
        "test_participants_json 未找到 locked_test_participants / "
        "test_participants / participant_ids"
    )


def load_test_sequences(
    features_dir: str,
    sex_tag: str,
    allowed_participants: set[str] | None = None,
):
    """
    从 features_{sex}.h5 直接读取 acquisition groups。

    返回一个 list；每个元素就是一个 acquisition-group 评估单元。
    评估单元为 acquisition group；原始 SeriesInstanceUID
    可能包含多个 acquisition group，因此两者不是同一个计数。
    同一 participant 可以有多次 examination，每次 examination 可以有任意数量期相，
    不要求每个患者或每次检查都同时包含四个类别。
    """
    h5_path = os.path.join(features_dir, f"features_{sex_tag}.h5")
    metadata_path = os.path.join(features_dir, f"metadata_{sex_tag}.pkl")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(metadata_path)

    with open(metadata_path, "rb") as file:
        metadata = pickle.load(file)

    validate_feature_metadata(metadata)

    examination_keys = metadata.get(
        "examination_keys",
        metadata.get("patient_ids", []),
    )

    rows = []
    skipped_by_participant_filter = 0

    with h5py.File(h5_path, "r") as hf:
        for exam_key in tqdm(examination_keys, desc=f"读取 {sex_tag} 测试集"):
            safe_key = str(exam_key).replace("/", "_")
            if safe_key not in hf:
                continue

            group = hf[safe_key]

            required = [
                "features",
                "sequence_ids",
                "sequence_labels",
                "sequence_timestamps",
                "sequence_seconds",
            ]
            missing = [name for name in required if name not in group]
            if missing:
                raise ValueError(f"{safe_key} 缺少字段: {missing}")

            participant_id = _decode(
                group.attrs.get("participant_id", "")
            ).strip()
            exam_id = _decode(group.attrs.get("exam_id", "")).strip()
            study_uid = _decode(group.attrs.get("study_uid", "")).strip()
            study_date = _decode(group.attrs.get("study_date", "")).strip()
            sex = _decode(group.attrs.get("sex", "U")).strip()

            if not participant_id:
                raise ValueError(f"{safe_key} 缺少 participant_id")

            if allowed_participants is not None and participant_id not in allowed_participants:
                skipped_by_participant_filter += 1
                continue

            features = group["features"][:].astype(np.float32, copy=False)
            sequence_ids = group["sequence_ids"][:].astype(np.int64)
            sequence_labels = group["sequence_labels"][:].astype(np.int64)
            sequence_timestamps = group["sequence_timestamps"][:].astype(np.float64)
            sequence_seconds = group["sequence_seconds"][:].astype(np.float64)

            sequence_series_uids = (
                [_decode(x) for x in group["sequence_series_uids"][:]]
                if "sequence_series_uids" in group
                else [""] * len(sequence_labels)
            )
            paths = (
                [_decode(x) for x in group["paths"][:]]
                if "paths" in group
                else [""] * len(features)
            )

            if len(features) != len(sequence_ids):
                raise ValueError(
                    f"{safe_key}: features={len(features)} 与 sequence_ids={len(sequence_ids)} 不一致"
                )

            for sequence_id in range(len(sequence_labels)):
                indices = np.flatnonzero(sequence_ids == sequence_id)
                if len(indices) == 0:
                    continue

                label = int(sequence_labels[sequence_id])
                if label < 0 or label >= NUM_CLASSES:
                    raise ValueError(f"{safe_key}: 非法标签 {label}")

                seconds = float(sequence_seconds[sequence_id])
                timestamp = float(sequence_timestamps[sequence_id])
                series_uid = (
                    sequence_series_uids[sequence_id]
                    if sequence_id < len(sequence_series_uids)
                    else ""
                )

                sequence_paths = [paths[i] for i in indices.tolist()]

                rows.append({
                    "participant_id": participant_id,
                    "exam_key": str(exam_key),
                    "exam_id": exam_id,
                    "study_uid": study_uid,
                    "study_date": study_date,
                    "sex": sex,
                    "sequence_id": int(sequence_id),
                    "evaluation_unit": "acquisition_group",
                    "series_uid": series_uid,
                    "timestamp": timestamp,
                    "acquisition_seconds": seconds,
                    "features": features[indices],
                    "paths": sequence_paths,
                    "label": label,
                    "gt_name": CLASS_NAMES[label],
                    "num_images": int(len(indices)),
                })

    # 检查内按真实采集时间排序；缺失时间时回退到 timestamp / sequence_id。
    def sort_key(item):
        seconds = item["acquisition_seconds"]
        if np.isfinite(seconds) and seconds >= 0:
            primary = seconds
        else:
            timestamp = item["timestamp"]
            primary = timestamp if np.isfinite(timestamp) and timestamp >= 0 else float("inf")
        return (
            item["participant_id"],
            item["exam_key"],
            primary,
            item["sequence_id"],
        )

    rows.sort(key=sort_key)

    print(
        f"  读取完成: {len(set(x['participant_id'] for x in rows))} 个患者, "
        f"{len(set(x['exam_key'] for x in rows))} 次检查, "
        f"{len(rows)} 个 acquisition groups"
    )
    if allowed_participants is not None:
        print(f"  participant filter 跳过 examination 数: {skipped_by_participant_filter}")

    counts = Counter(x["gt_name"] for x in rows)
    print("  GT类别分布:", {name: counts.get(name, 0) for name in CLASS_NAMES})

    return rows, metadata


def make_fixed_length_input(features: np.ndarray):
    """推理时把一个 acquisition group 固定为 [20, D]，并生成 padding mask。"""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or len(features) == 0:
        raise ValueError(f"非法 sequence features shape={features.shape}")

    real_length = len(features)
    feature_dim = features.shape[1]

    if real_length > SEQUENCE_LENGTH:
        indices = np.linspace(
            0,
            real_length - 1,
            SEQUENCE_LENGTH,
            dtype=np.int64,
        )
        selected = features[indices]
        valid_length = SEQUENCE_LENGTH
    else:
        selected = features
        valid_length = real_length

    padded = np.zeros(
        (SEQUENCE_LENGTH, feature_dim),
        dtype=np.float32,
    )
    padded[:valid_length] = selected[:valid_length]

    attention_mask = np.ones(SEQUENCE_LENGTH, dtype=np.bool_)
    attention_mask[:valid_length] = False

    return padded, attention_mask, valid_length


# ============================================================================
# 5. 五折概率集成
# ============================================================================

@torch.inference_mode()
def predict_rows(
    rows: list[dict],
    models: list[nn.Module],
    device: torch.device,
    use_amp: bool = False,
):
    """
    每个 fold 输出四类 softmax 概率；
    对五折概率向量逐类别取算术平均后 argmax，得到 Category_Raw。
    """
    amp_enabled = bool(use_amp and device.type == "cuda")

    for row in tqdm(rows, desc="5-fold probability ensemble"):
        padded, attention_mask, valid_length = make_fixed_length_input(
            row["features"]
        )

        x = torch.from_numpy(padded).unsqueeze(0).to(device)
        mask = torch.from_numpy(attention_mask).unsqueeze(0).to(device)

        fold_probabilities = []

        for model in models:
            if amp_enabled:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    logits = model(x, attention_mask=mask)
            else:
                logits = model(x, attention_mask=mask)

            probabilities = F.softmax(logits, dim=1)[0]
            fold_probabilities.append(
                probabilities.float().cpu().numpy()
            )

        fold_probabilities = np.stack(fold_probabilities, axis=0)
        mean_probabilities = fold_probabilities.mean(axis=0)
        raw_index = int(np.argmax(mean_probabilities))

        row["valid_length"] = int(valid_length)
        row["fold_probabilities"] = fold_probabilities.tolist()
        row["mean_probabilities"] = mean_probabilities.tolist()
        row["raw_index"] = raw_index
        row["Category_Raw"] = CLASS_NAMES[raw_index]

    return rows


@torch.inference_mode()
def predict_rows_routed(
    rows: list[dict],
    models_by_sex: dict[str, list[nn.Module]],
    device: torch.device,
    use_amp: bool = False,
):
    """按 DICOM PatientSex 路由到对应的性别模型，再做五折概率集成。

    可联合评估男女样本；male/female 单独运行
    仍由 ``predict_rows`` 保持兼容。未知或缺失 PatientSex 会立即报错，避免
    把未定义路由样本静默混入汇总结果。
    """
    for row in tqdm(rows, desc="routed 5-fold probability ensemble"):
        sex = str(row.get("sex", "")).strip().upper()
        sex_tag = {"M": "male", "F": "female"}.get(sex)
        if sex_tag is None:
            raise ValueError(
                "combined routed evaluation 遇到缺失或未知 PatientSex："
                f"participant={row.get('participant_id')} sex={sex!r}"
            )

        models = models_by_sex[sex_tag]
        padded, attention_mask, valid_length = make_fixed_length_input(
            row["features"]
        )
        x = torch.from_numpy(padded).unsqueeze(0).to(device)
        mask = torch.from_numpy(attention_mask).unsqueeze(0).to(device)

        fold_probabilities = []
        for model in models:
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x, attention_mask=mask)
            else:
                logits = model(x, attention_mask=mask)
            fold_probabilities.append(
                F.softmax(logits, dim=1)[0].float().cpu().numpy()
            )

        fold_probabilities = np.stack(fold_probabilities, axis=0)
        mean_probabilities = fold_probabilities.mean(axis=0)
        raw_index = int(np.argmax(mean_probabilities))

        row["routed_model"] = sex_tag
        row["valid_length"] = int(valid_length)
        row["fold_probabilities"] = fold_probabilities.tolist()
        row["mean_probabilities"] = mean_probabilities.tolist()
        row["raw_index"] = raw_index
        row["Category_Raw"] = CLASS_NAMES[raw_index]

    return rows


# ============================================================================
# 6. Category_Raw -> Category_Predict 后处理
# ============================================================================

def copy_raw_as_unavailable_category_predict(rows: list[dict]):
    """在确切后处理规则不可得时，建立明确标记的占位输出。

    在尚未确定后处理规则时，将 Raw 暂存为 Predict 以保持输出字段完整，
    并显式标记该模式。此时的 Predict 不表示已经执行了真实顺序修正。
    """
    for row in rows:
        row["Category_Predict"] = row["Category_Raw"]
        row["predict_index"] = int(row["raw_index"])
        row["Order_Corrected"] = False
    return []


def apply_reconstructed_ordering_rule_to_exam(
    exam_rows: list[dict],
    enabled: bool = True,
):
    """
    诊断性重建的期相顺序性逻辑：

    可选的顺序约束规则（与默认占位模式相互独立）：
      precontrast -> arterial -> venous -> delayed

    - precontrast 被视为一次新的动态增强序列起点，直接接受并将状态重置到0；
    - 除 precontrast 外，其余类别只能保持当前阶段或向前；
    - 如果 raw 预测发生回退，则强制改为当前已经到达的最高阶段。

    例如：
      raw: delayed -> venous
      final: delayed -> delayed
    此时第二个 venous 会被改为 delayed。
    """
    if not exam_rows:
        return []

    corrections = []
    current_phase_index = 0

    for row in exam_rows:
        raw_name = row["Category_Raw"]
        raw_index = PHASE_ORDER[raw_name]

        final_name = raw_name

        if enabled:
            if raw_name == "precontrast":
                current_phase_index = 0
            elif raw_index >= current_phase_index:
                current_phase_index = raw_index
            else:
                final_name = CLASS_NAMES[current_phase_index]
                corrections.append({
                    "participant_id": row["participant_id"],
                    "exam_key": row["exam_key"],
                    "sequence_id": row["sequence_id"],
                    "original": raw_name,
                    "corrected": final_name,
                    "reason": (
                        f"非法回退: raw_index={raw_index} < "
                        f"current_phase_index={current_phase_index}"
                    ),
                })

        row["Category_Predict"] = final_name
        row["predict_index"] = PHASE_ORDER[final_name]
        row["Order_Corrected"] = final_name != raw_name

    return corrections


def apply_reconstructed_ordering_rule(rows: list[dict], enabled: bool = True):
    """逐 examination 应用未验证的顺序规则，不能跨 examination 传递状态。"""
    by_exam = defaultdict(list)
    for row in rows:
        by_exam[row["exam_key"]].append(row)

    all_corrections = []

    for exam_key, exam_rows in by_exam.items():
        exam_rows.sort(
            key=lambda item: (
                item["acquisition_seconds"]
                if np.isfinite(item["acquisition_seconds"])
                and item["acquisition_seconds"] >= 0
                else float("inf"),
                item["sequence_id"],
            )
        )
        all_corrections.extend(
            apply_reconstructed_ordering_rule_to_exam(exam_rows, enabled=enabled)
        )

    return all_corrections


# ============================================================================
# 7. 指标
# ============================================================================

def _safe_specificity(y_true, y_pred, class_index: int) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    negative = y_true != class_index
    denominator = int(negative.sum())
    if denominator == 0:
        return float("nan")

    true_negative = int(((y_true != class_index) & (y_pred != class_index)).sum())
    return true_negative / denominator


def classification_metrics(y_true, y_pred):
    """计算 acquisition-group 级的四分类指标。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    labels = list(range(NUM_CLASSES))

    accuracy = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels))

    macro_precision = float(
        precision_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    macro_recall = float(
        recall_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision = float(
        precision_score(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        )
    )
    weighted_recall = float(
        recall_score(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        )
    )
    weighted_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        )
    )

    class_precision = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    class_recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    class_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    per_class = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        support = int((y_true == class_index).sum())
        predicted_positive = int((y_pred == class_index).sum())
        true_positive = int(((y_true == class_index) & (y_pred == class_index)).sum())

        per_class[class_name] = {
            "precision": float(class_precision[class_index]),
            "recall": float(class_recall[class_index]),
            "specificity": float(
                _safe_specificity(y_true, y_pred, class_index)
            ),
            "f1": float(class_f1[class_index]),
            "support": support,
            "predicted_positive": predicted_positive,
            "true_positive": true_positive,
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "n": int(len(y_true)),
        "accuracy": accuracy,
        "kappa": kappa,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def examination_metrics(rows: list[dict], prediction_key: str):
    """
    Examination accuracy = 每次检查中预测正确的 acquisition-group 比例；
    complete examination correctness = 该次检查全部 acquisition group 都正确。
    """
    by_exam = defaultdict(list)
    for row in rows:
        by_exam[row["exam_key"]].append(row)

    details = []
    for exam_key, exam_rows in by_exam.items():
        correct = [
            int(row[prediction_key] == row["gt_name"])
            for row in exam_rows
        ]
        accuracy = float(np.mean(correct)) if correct else float("nan")
        complete = bool(all(correct)) if correct else False

        first = exam_rows[0]
        details.append({
            "participant_id": first["participant_id"],
            "exam_key": exam_key,
            "exam_id": first["exam_id"],
            "study_uid": first["study_uid"],
            "study_date": first["study_date"],
            "sex": first["sex"],
            "n_evaluation_units": len(exam_rows),
            # 保留旧键名，避免下游脚本失效；该值实际是 acquisition-group 数。
            "n_series": len(exam_rows),
            "n_correct": int(sum(correct)),
            "accuracy": accuracy,
            "complete_correct": complete,
        })

    accuracies = np.asarray([x["accuracy"] for x in details], dtype=float)
    complete_count = int(sum(x["complete_correct"] for x in details))

    return {
        "n_examinations": len(details),
        "mean_accuracy": float(np.nanmean(accuracies)) if len(accuracies) else float("nan"),
        "std_accuracy": float(np.nanstd(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
        "median_accuracy": float(np.nanmedian(accuracies)) if len(accuracies) else float("nan"),
        "q1_accuracy": float(np.nanpercentile(accuracies, 25)) if len(accuracies) else float("nan"),
        "q3_accuracy": float(np.nanpercentile(accuracies, 75)) if len(accuracies) else float("nan"),
        "complete_correct_count": complete_count,
        "complete_correct_rate": (
            complete_count / len(details)
            if details
            else float("nan")
        ),
        "details": details,
    }


def clopper_pearson_interval(successes: int, trials: int, alpha: float = 0.05):
    """Clopper-Pearson exact binomial confidence interval。"""
    if trials <= 0:
        return [float("nan"), float("nan")]

    if beta_distribution is None:
        raise RuntimeError(
            "计算 exact binomial CI 需要 scipy。请安装 scipy 后重新运行。"
        )

    if successes == 0:
        lower = 0.0
    else:
        lower = float(
            beta_distribution.ppf(
                alpha / 2,
                successes,
                trials - successes + 1,
            )
        )

    if successes == trials:
        upper = 1.0
    else:
        upper = float(
            beta_distribution.ppf(
                1 - alpha / 2,
                successes + 1,
                trials - successes,
            )
        )

    return [lower, upper]


# ============================================================================
# 8. Participant-clustered bootstrap
# ============================================================================

def _percentile_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [float("nan"), float("nan")]
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def clustered_bootstrap(
    rows: list[dict],
    prediction_key: str,
    n_iterations: int = BOOTSTRAP_ITERATIONS_DEFAULT,
    seed: int = BOOTSTRAP_SEED_DEFAULT,
):
    """按 participant 聚类重采样，其全部 examination/acquisition groups 一起进入样本。"""
    participant_rows = defaultdict(list)
    for row in rows:
        participant_rows[row["participant_id"]].append(row)

    participant_ids = np.asarray(sorted(participant_rows.keys()), dtype=object)
    if len(participant_ids) == 0:
        return {}

    rng = np.random.default_rng(seed)

    metric_samples = defaultdict(list)
    per_class_samples = {
        class_name: defaultdict(list)
        for class_name in CLASS_NAMES
    }

    exam_mean_accuracy = []
    exam_complete_rate = []

    for _ in tqdm(range(n_iterations), desc="participant-clustered bootstrap"):
        sampled_ids = rng.choice(
            participant_ids,
            size=len(participant_ids),
            replace=True,
        )

        sampled_rows = []
        # 重复抽到同一 participant 时，其全部 acquisition groups 也重复进入样本。
        for draw_index, participant_id in enumerate(sampled_ids.tolist()):
            for row in participant_rows[participant_id]:
                copied = dict(row)
                # exam_key 加 draw_index，防止同一 participant 重复抽样时被合并成同一次检查。
                copied["exam_key"] = f"boot{draw_index}::{row['exam_key']}"
                sampled_rows.append(copied)

        y_true = [row["label"] for row in sampled_rows]
        y_pred = [PHASE_ORDER[row[prediction_key]] for row in sampled_rows]

        metrics = classification_metrics(y_true, y_pred)
        for key in (
            "accuracy",
            "kappa",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
        ):
            metric_samples[key].append(metrics[key])

        for class_name in CLASS_NAMES:
            for key in ("precision", "recall", "specificity", "f1"):
                per_class_samples[class_name][key].append(
                    metrics["per_class"][class_name][key]
                )

        exam_metrics = examination_metrics(sampled_rows, prediction_key)
        exam_mean_accuracy.append(exam_metrics["mean_accuracy"])
        exam_complete_rate.append(exam_metrics["complete_correct_rate"])

    return {
        "iterations": int(n_iterations),
        "seed": int(seed),
        "series_metrics_95ci": {
            key: _percentile_ci(values)
            for key, values in metric_samples.items()
        },
        "per_class_95ci": {
            class_name: {
                key: _percentile_ci(values)
                for key, values in class_metrics.items()
            }
            for class_name, class_metrics in per_class_samples.items()
        },
        "examination_mean_accuracy_95ci": _percentile_ci(exam_mean_accuracy),
        "complete_examination_rate_95ci": _percentile_ci(exam_complete_rate),
    }


# ============================================================================
# 9. 汇总
# ============================================================================

def evaluate_prediction_set(
    rows: list[dict],
    prediction_key: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
):
    y_true = [row["label"] for row in rows]
    y_pred = [PHASE_ORDER[row[prediction_key]] for row in rows]

    series = classification_metrics(y_true, y_pred)
    exam = examination_metrics(rows, prediction_key)
    bootstrap = clustered_bootstrap(
        rows,
        prediction_key,
        n_iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )

    delayed = series["per_class"]["delayed"]
    delayed_exact_ci = clopper_pearson_interval(
        delayed["true_positive"],
        delayed["predicted_positive"],
    )

    series["per_class"]["delayed"]["precision_exact_binomial_95ci"] = delayed_exact_ci

    return {
        "evaluation_unit_level": series,
        # 保留既有输出字段；每行对应一个 acquisition group。
        "series_level": series,
        "examination_level": exam,
        "bootstrap": bootstrap,
    }


def build_output_summary(
    rows: list[dict],
    corrections: list[dict],
    sex_tag: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    postprocessing_mode: str,
):
    raw = evaluate_prediction_set(
        rows,
        "Category_Raw",
        bootstrap_iterations,
        bootstrap_seed,
    )
    final = evaluate_prediction_set(
        rows,
        "Category_Predict",
        bootstrap_iterations,
        bootstrap_seed,
    )

    transition_counts = Counter(
        f"{item['original']}->{item['corrected']}"
        for item in corrections
    )

    if postprocessing_mode == "unavailable":
        postprocessing_status = "unavailable_in_study_documentation"
        postprocessing_rule = None
        postprocessing_note = (
            "Category_Predict is a raw-output placeholder because the study's "
            "exact postprocessing rule, thresholds, and prespecification are unavailable."
        )
    elif postprocessing_mode == "reconstructed_ordering":
        postprocessing_status = "reconstructed_unverified"
        postprocessing_rule = (
            "nondecreasing phase order within examination; precontrast resets state"
        )
        postprocessing_note = (
            "This rule is a diagnostic reconstruction and must not be presented "
            "as the study's documented algorithm."
        )
    else:
        raise ValueError(f"未知 postprocessing_mode={postprocessing_mode!r}")

    source_series = {
        (row["exam_key"], row.get("series_uid", ""))
        for row in rows
    }

    return {
        "sex_tag": sex_tag,
        "n_participants": len(set(row["participant_id"] for row in rows)),
        "n_examinations": len(set(row["exam_key"] for row in rows)),
        "n_evaluation_units": len(rows),
        "evaluation_unit": "acquisition_group",
        "manuscript_unit_label": "reported series",
        "n_source_series": len(source_series),
        # 向后兼容；不要将其解释为 distinct SeriesInstanceUID 计数。
        "n_series": len(rows),
        "class_names": CLASS_NAMES,
        "ensemble": "arithmetic mean of five fold-specific class-probability vectors, then argmax",
        "sequence_length": SEQUENCE_LENGTH,
        "category_raw": raw,
        "category_predict": final,
        "postprocessing": {
            "mode": postprocessing_mode,
            "status": postprocessing_status,
            "publication_usable": False,
            "rule": postprocessing_rule,
            "note": postprocessing_note,
            "phase_order": CLASS_NAMES,
            "n_corrections": len(corrections),
            "transition_counts": dict(transition_counts),
            "corrections": corrections,
        },
    }


# ============================================================================
# 10. 输出文件
# ============================================================================

def save_series_csv(rows: list[dict], output_path: str):
    probability_columns = [f"P_{name}" for name in CLASS_NAMES]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "participant_id",
            "exam_key",
            "exam_id",
            "study_uid",
            "study_date",
            "sex",
            "evaluation_unit",
            "sequence_id",
            "series_uid",
            "acquisition_seconds",
            "timestamp",
            "num_images",
            "GT",
            "Category_Raw",
            "Category_Predict",
            "Order_Corrected",
            *probability_columns,
        ])

        for row in rows:
            writer.writerow([
                row["participant_id"],
                row["exam_key"],
                row["exam_id"],
                row["study_uid"],
                row["study_date"],
                row["sex"],
                row.get("evaluation_unit", "acquisition_group"),
                row["sequence_id"],
                row["series_uid"],
                row["acquisition_seconds"],
                row["timestamp"],
                row["num_images"],
                row["gt_name"],
                row["Category_Raw"],
                row["Category_Predict"],
                "是" if row["Order_Corrected"] else "否",
                *[float(x) for x in row["mean_probabilities"]],
            ])


def save_image_csv(rows: list[dict], output_path: str):
    """把 acquisition-group 预测广播到组内所有 DICOM 图像，便于全档案标签导出。"""
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "participant_id",
            "exam_key",
            "exam_id",
            "study_uid",
            "study_date",
            "evaluation_unit",
            "sequence_id",
            "series_uid",
            "DICOM_Path",
            "GT",
            "Category_Raw",
            "Category_Predict",
            "Order_Corrected",
        ])

        for row in rows:
            for path in row.get("paths", []):
                writer.writerow([
                    row["participant_id"],
                    row["exam_key"],
                    row["exam_id"],
                    row["study_uid"],
                    row["study_date"],
                    row.get("evaluation_unit", "acquisition_group"),
                    row["sequence_id"],
                    row["series_uid"],
                    path,
                    row["gt_name"],
                    row["Category_Raw"],
                    row["Category_Predict"],
                    "是" if row["Order_Corrected"] else "否",
                ])


def save_report(summary: dict, output_path: str):
    """保存人类可读报告。"""
    raw = summary["category_raw"]
    final = summary["category_predict"]

    def pct(value):
        return f"{100.0 * value:.2f}%" if np.isfinite(value) else "NA"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("四期相 5-fold probability ensemble 评估报告\n")
        file.write("=" * 72 + "\n")
        file.write(
            f"participants={summary['n_participants']}, "
            f"examinations={summary['n_examinations']}, "
            f"evaluation_units(acquisition_groups)={summary['n_evaluation_units']}, "
            f"source_series={summary['n_source_series']}\n"
        )
        file.write(
            "稿件报告术语: reported series；代码评估单元: acquisition group\n"
        )
        file.write(f"sex={summary['sex_tag']}\n")
        file.write(f"ensemble={summary['ensemble']}\n")
        file.write("\n")

        for title, result in (
            ("Category_Raw（五折集成结果）", raw),
            ("Category_Predict（后处理输出/占位）", final),
        ):
            series = result["series_level"]
            exam = result["examination_level"]
            ci = result["bootstrap"]["series_metrics_95ci"]

            file.write(title + "\n")
            file.write("-" * 72 + "\n")
            file.write(
                f"Evaluation-unit accuracy: {pct(series['accuracy'])} "
                f"(95% CI {pct(ci['accuracy'][0])}-{pct(ci['accuracy'][1])})\n"
            )
            file.write(
                f"Cohen kappa: {series['kappa']:.4f} "
                f"(95% CI {ci['kappa'][0]:.4f}-{ci['kappa'][1]:.4f})\n"
            )
            file.write(f"Macro precision: {pct(series['macro_precision'])}\n")
            file.write(f"Macro recall: {pct(series['macro_recall'])}\n")
            file.write(f"Macro F1: {pct(series['macro_f1'])}\n")
            file.write(f"Weighted precision: {pct(series['weighted_precision'])}\n")
            file.write(f"Weighted recall: {pct(series['weighted_recall'])}\n")
            file.write(f"Weighted F1: {pct(series['weighted_f1'])}\n")
            file.write(
                f"Examination mean accuracy: {pct(exam['mean_accuracy'])} "
                f"± {pct(exam['std_accuracy'])}\n"
            )
            file.write(
                f"Complete examinations: {exam['complete_correct_count']}/"
                f"{exam['n_examinations']} ({pct(exam['complete_correct_rate'])})\n"
            )

            file.write("Per-class:\n")
            for class_name in CLASS_NAMES:
                m = series["per_class"][class_name]
                file.write(
                    f"  {class_name}: support={m['support']}, "
                    f"precision={pct(m['precision'])}, "
                    f"recall={pct(m['recall'])}, "
                    f"specificity={pct(m['specificity'])}, "
                    f"F1={pct(m['f1'])}\n"
                )
            file.write("\n")

        post = summary["postprocessing"]
        file.write("后处理\n")
        file.write("-" * 72 + "\n")
        file.write(f"模式: {post['mode']}\n")
        file.write(f"状态: {post['status']}\n")
        file.write(f"可直接用于稿件主分析: {'是' if post['publication_usable'] else '否'}\n")
        file.write(f"说明: {post['note']}\n")
        file.write(f"修正次数: {post['n_corrections']}\n")
        file.write(f"修正方向: {post['transition_counts']}\n")

        raw_acc = raw["series_level"]["accuracy"]
        final_acc = final["series_level"]["accuracy"]
        file.write(
            f"Category_Raw accuracy={pct(raw_acc)}, "
            f"Category_Predict accuracy={pct(final_acc)}, "
            f"difference={100.0 * (final_acc - raw_acc):.2f} percentage points\n"
        )


# ============================================================================
# 11. 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="按当前 Methods 进行四期相 5-fold ensemble 推理与评估"
    )

    parser.add_argument(
        "--model_dir",
        default=None,
        help="male/female 单性别模式下包含 fold_0~fold_4 的模型目录",
    )
    parser.add_argument(
        "--features_dir",
        default=None,
        help="male/female 单性别模式下的 locked internal test 特征目录",
    )
    parser.add_argument(
        "--sex_tag",
        required=True,
        choices=["male", "female", "all"],
        help="male/female 单性别评估，或 all=combined routed pipeline",
    )
    parser.add_argument(
        "--model_dir_male",
        default=None,
        help="sex_tag=all 时的 male fold 模型目录",
    )
    parser.add_argument(
        "--features_dir_male",
        default=None,
        help="sex_tag=all 时的 male 测试特征目录",
    )
    parser.add_argument(
        "--model_dir_female",
        default=None,
        help="sex_tag=all 时的 female fold 模型目录",
    )
    parser.add_argument(
        "--features_dir_female",
        default=None,
        help="sex_tag=all 时的 female 测试特征目录",
    )
    parser.add_argument(
        "--output_dir",
        default="./eval_phase_kfold_methods",
    )
    parser.add_argument(
        "--test_participants_json",
        default=None,
        help="可选：locked internal test participant ID 列表；不传则评估 features_dir 中全部患者",
    )
    parser.add_argument(
        "--postprocessing_mode",
        choices=["unavailable", "reconstructed_ordering"],
        default="unavailable",
        help=(
            "稿件未提供确切后处理规则；默认 unavailable 并将 Category_Predict "
            "标记为 Raw 占位副本。reconstructed_ordering 仅用于明确标记的诊断性重建。"
        ),
    )
    parser.add_argument(
        "--no_order_constraint",
        action="store_true",
        help=(
            "兼容旧命令行：等价于 --postprocessing_mode unavailable；"
            "不能用于声明稿件的最终后处理。"
        ),
    )
    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=BOOTSTRAP_ITERATIONS_DEFAULT,
        help="participant-clustered bootstrap 次数，Methods 默认10000",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=BOOTSTRAP_SEED_DEFAULT,
        help="bootstrap随机种子，Methods 默认20260922",
    )
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="显式开启 CUDA AMP 推理；默认FP32",
    )

    args = parser.parse_args()

    if args.no_order_constraint:
        if args.postprocessing_mode == "reconstructed_ordering":
            raise ValueError(
                "--no_order_constraint 与 --postprocessing_mode "
                "reconstructed_ordering 不能同时使用"
            )
        args.postprocessing_mode = "unavailable"

    random.seed(args.bootstrap_seed)
    np.random.seed(args.bootstrap_seed)
    torch.manual_seed(args.bootstrap_seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        f"cuda:{args.device_id}"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("四期相 5-fold probability ensemble")
    print(f"sex={args.sex_tag} | device={device}")
    print("classes=precontrast/arterial/venous/delayed")
    print("fixed sequence length=20")
    print("fold ensemble=逐类别softmax概率算术平均 -> argmax")
    print(f"postprocessing mode={args.postprocessing_mode}")
    print("=" * 80)

    print("\n[1] 加载5折四分类模型")
    allowed_participants = load_allowed_participants(
        args.test_participants_json
    )

    if args.sex_tag == "all":
        required_all = {
            "model_dir_male": args.model_dir_male,
            "features_dir_male": args.features_dir_male,
            "model_dir_female": args.model_dir_female,
            "features_dir_female": args.features_dir_female,
        }
        missing_all = [key for key, value in required_all.items() if not value]
        if missing_all:
            raise ValueError(
                "sex_tag=all 必须同时提供: " + ", ".join(missing_all)
            )

        male_models, male_checkpoints = load_kfold_models(
            args.model_dir_male,
            device,
        )
        female_models, female_checkpoints = load_kfold_models(
            args.model_dir_female,
            device,
        )

        print("\n[2] 读取 male/female 测试特征并按 PatientSex 路由")
        male_rows, male_metadata = load_test_sequences(
            features_dir=args.features_dir_male,
            sex_tag="male",
            allowed_participants=allowed_participants,
        )
        female_rows, female_metadata = load_test_sequences(
            features_dir=args.features_dir_female,
            sex_tag="female",
            allowed_participants=allowed_participants,
        )

        male_participants = {row["participant_id"] for row in male_rows}
        female_participants = {row["participant_id"] for row in female_rows}
        overlap = male_participants & female_participants
        if overlap:
            raise ValueError(
                "male/female 测试特征出现相同 participant_id，无法安全路由："
                f"{sorted(overlap)[:10]}"
            )

        rows = male_rows + female_rows
        metadata = {
            "male": male_metadata,
            "female": female_metadata,
        }
        models_by_sex = {
            "male": male_models,
            "female": female_models,
        }
        model_dirs = {
            "male": os.path.abspath(args.model_dir_male),
            "female": os.path.abspath(args.model_dir_female),
        }
        features_dirs = {
            "male": os.path.abspath(args.features_dir_male),
            "female": os.path.abspath(args.features_dir_female),
        }
        output_sex_tag = "routed"
    else:
        if not args.model_dir or not args.features_dir:
            raise ValueError(
                "sex_tag=male/female 必须提供 --model_dir 和 --features_dir"
            )
        models, checkpoints = load_kfold_models(args.model_dir, device)
        print("\n[2] 读取测试特征")
        rows, metadata = load_test_sequences(
            features_dir=args.features_dir,
            sex_tag=args.sex_tag,
            allowed_participants=allowed_participants,
        )
        model_dirs = os.path.abspath(args.model_dir)
        features_dirs = os.path.abspath(args.features_dir)
        output_sex_tag = args.sex_tag

    if not rows:
        raise RuntimeError("测试集没有可评估的 acquisition groups")

    print("\n[3] 5-fold 概率集成推理")
    if args.sex_tag == "all":
        rows = predict_rows_routed(
            rows,
            models_by_sex,
            device,
            use_amp=args.amp,
        )
    else:
        rows = predict_rows(rows, models, device, use_amp=args.amp)

    print("\n[4] Category_Raw -> Category_Predict 后处理")
    if args.postprocessing_mode == "reconstructed_ordering":
        corrections = apply_reconstructed_ordering_rule(rows, enabled=True)
    else:
        corrections = copy_raw_as_unavailable_category_predict(rows)
    print(
        f"  模式={args.postprocessing_mode} | "
        f"修正次数={len(corrections)}"
    )
    if corrections:
        transition_counts = Counter(
            f"{x['original']}->{x['corrected']}"
            for x in corrections
        )
        print(f"  修正方向: {dict(transition_counts)}")

    print("\n[5] 计算指标与 participant-clustered bootstrap")
    summary = build_output_summary(
        rows=rows,
        corrections=corrections,
        sex_tag=output_sex_tag,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        postprocessing_mode=args.postprocessing_mode,
    )

    summary["model_dir"] = model_dirs
    summary["features_dir"] = features_dirs
    summary["feature_preprocessing"] = (
        {
            "male": metadata["male"].get("preprocessing", {}),
            "female": metadata["female"].get("preprocessing", {}),
        }
        if args.sex_tag == "all"
        else metadata.get("preprocessing", {})
    )
    summary["bootstrap_iterations"] = args.bootstrap_iterations
    summary["bootstrap_seed"] = args.bootstrap_seed
    summary["routing"] = (
        "PatientSex: M->male model, F->female model"
        if args.sex_tag == "all"
        else f"single sex model: {args.sex_tag}"
    )

    json_path = os.path.join(args.output_dir, "raw_diagnosis.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    series_csv_path = os.path.join(
        args.output_dir,
        f"series_predictions_{output_sex_tag}.csv",
    )
    save_series_csv(rows, series_csv_path)

    image_csv_path = os.path.join(
        args.output_dir,
        f"predicted_category_{output_sex_tag}.csv",
    )
    save_image_csv(rows, image_csv_path)

    report_path = os.path.join(args.output_dir, "raw_diagnosis.txt")
    save_report(summary, report_path)

    print("\n[6] 完成")
    print(f"  JSON: {json_path}")
    print(f"  Series CSV: {series_csv_path}")
    print(f"  Image CSV: {image_csv_path}")
    print(f"  Report: {report_path}")

    raw_acc = summary["category_raw"]["series_level"]["accuracy"]
    final_acc = summary["category_predict"]["series_level"]["accuracy"]
    print(
        f"  Category_Raw accuracy={raw_acc * 100:.2f}% | "
        f"Category_Predict accuracy={final_acc * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
