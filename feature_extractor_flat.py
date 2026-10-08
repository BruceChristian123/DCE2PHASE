#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConvNeXt-Base 图像特征提取

输入：
    patient_cache_{sex}.pkl

输出：
    features_{sex}.h5
    metadata_{sex}.pkl

图像处理流程：
    DICOM 像素 → 逐图像 Min-Max → 短边缩放至 256 → 中心裁剪 224×224
    → 复制为三通道 → ImageNet 归一化 → 冻结的 ConvNeXt-Base

每张有效图像提取 1024 维特征。HDF5 按 examination 存储，保留
图像级 sequence_ids、原始路径、时间和序列 UID，以及各 acquisition group
的标签、图像数量和时间信息，避免训练阶段再次按时间戳分组。

HDF5 examination group：
    features               [N, 1024]
    labels                 [N]
    sequence_ids           [N]
    timestamps             [N]
    acquisition_seconds    [N]
    series_uids            [N]
    paths                  [N]
    sequence_labels        [S]
    sequence_timestamps    [S]
    sequence_seconds       [S]
    sequence_lengths       [S]
    sequence_series_uids   [S]

participant_id 保存在 examination 属性中。同一患者的多次检查应在
后续数据划分时保持在同一个 fold。
"""

import argparse
import os
import pickle
import warnings
from contextlib import nullcontext
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

warnings.filterwarnings("ignore")


# 统一使用以下图像尺寸和插值参数。
# torchvision 的 ConvNeXt 权重 preset 在不同版本中可能使用 236 的短边尺寸，
# 为避免预训练权重配置差异改变输入尺寸，这里显式指定尺寸。
METHODS_RESIZE_SHORT_SIDE = 256
METHODS_CROP_SIZE = 224
METHODS_INTERPOLATION = InterpolationMode.BILINEAR
METHODS_FEATURE_SCHEMA_VERSION = (
    "methods_2026_v4_sequence_boundary_0p5s_spatial256_bilinear"
)


# ============================================================================
# 1. 图像预处理
# ============================================================================

def convert_to_2d(image: np.ndarray) -> np.ndarray:
    """
    将 DICOM pixel_array 整理为二维灰度图。

    主要处理二维 MRI 切片。遇到带有额外维度的数据时，采用以下兼容处理：
    - 通道在最后且通道数 <= 4：取灰度或转换为灰度；
    - 通道在最前且通道数 <= 4：取第一个通道；
    - 其他额外维度：逐层取第一个元素，直到变成二维。

    这里只负责维度整理，不进行强度归一化。
    """
    image = np.asarray(image)

    if image.ndim == 3:
        if image.shape[2] <= 4:
            if image.shape[2] == 1:
                image = image[:, :, 0]
            elif image.shape[2] >= 3:
                image = np.dot(
                    image[..., :3],
                    np.array([0.299, 0.587, 0.114], dtype=np.float32),
                )
            else:
                image = image[:, :, 0]
        elif image.shape[0] <= 4:
            image = image[0, :, :]
        else:
            image = image[:, :, 0]

    while image.ndim > 2:
        image = image[0]

    if image.ndim != 2:
        raise ValueError(f"无法转换为二维图像，shape={image.shape}")

    return image.astype(np.float32, copy=False)


def minmax_normalize(image: np.ndarray) -> np.ndarray:
    """
    对每张图像独立进行 min-max normalization，输出 float32 的 [0, 1]。
    """
    image = image.astype(np.float32, copy=False)

    finite_mask = np.isfinite(image)
    if not finite_mask.any():
        return np.zeros_like(image, dtype=np.float32)

    # 极少数异常像素若为 NaN/Inf，用有限像素中的最小值替代，避免传播 NaN。
    finite_values = image[finite_mask]
    vmin = float(finite_values.min())
    vmax = float(finite_values.max())

    if vmax - vmin < 1e-8:
        return np.zeros_like(image, dtype=np.float32)

    image = np.where(finite_mask, image, vmin)
    image = (image - vmin) / (vmax - vmin)

    # 显式映射到 0–255 后再还原到 [0, 1]；不转换为 uint8，
    # 避免额外量化误差；数值上仍等价于逐图像 min-max 到 [0, 1]。
    image = np.clip(image * 255.0, 0.0, 255.0) / 255.0
    return image.astype(np.float32, copy=False)


def _extract_single_size(value, name: str) -> int:
    """从 torchvision 权重预处理配置中取得单个尺寸。"""
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return int(value[0])
        if len(value) == 2 and value[0] == value[1]:
            return int(value[0])
    if isinstance(value, int):
        return int(value)
    raise ValueError(f"无法解析 {name}: {value}")


def get_pretrained_preprocess_config() -> Dict:
    """
    读取预训练权重的归一化配置，并使用固定的
    256→224 空间预处理参数。
    """
    weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
    preset = weights.transforms()

    # 从权重配置读取 mean/std，空间尺寸和插值采用脚本中的固定参数。
    resize_size = METHODS_RESIZE_SHORT_SIDE
    crop_size = METHODS_CROP_SIZE

    # ConvNeXt ImageNet preset 在不同 torchvision 版本中可能使用 bicubic；
    # 统一使用 bilinear，避免 torchvision 版本差异影响结果。
    interpolation = METHODS_INTERPOLATION
    mean = list(getattr(preset, "mean", [0.485, 0.456, 0.406]))
    std = list(getattr(preset, "std", [0.229, 0.224, 0.225]))

    return {
        "weights": weights,
        "weights_name": "ConvNeXt_Base_Weights.IMAGENET1K_V1",
        "resize_size": resize_size,
        "crop_size": crop_size,
        "resize_rule": "short_side_256_then_center_crop_224",
        "interpolation_name": "bilinear",
        "interpolation": interpolation,
        "mean": mean,
        "std": std,
    }


class DicomDataset(Dataset):
    """读取 DICOM 图像，完成归一化、缩放和中心裁剪。"""

    def __init__(
        self,
        image_paths: List[str],
        resize_size: int,
        crop_size: int,
        mean: List[float],
        std: List[float],
        interpolation=InterpolationMode.BILINEAR,
    ):
        self.paths = image_paths
        self.resize_size = int(resize_size)
        self.crop_size = int(crop_size)
        self.interpolation = interpolation
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        try:
            # 读取像素数据。
            dcm = pydicom.dcmread(path, force=True)
            image = convert_to_2d(dcm.pixel_array)

            # 每张图像单独进行 min-max normalization。
            image = minmax_normalize(image)

            # 单通道复制为三通道。
            image = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)

            # resize 后 center crop。
            # torchvision 对整数 resize_size 的含义是把短边缩放到指定尺寸，
            # 保持原始长宽比，然后再进行中心裁剪。
            image = TF.resize(
                image,
                size=self.resize_size,
                interpolation=self.interpolation,
                antialias=True,
            )
            image = TF.center_crop(
                image,
                output_size=[self.crop_size, self.crop_size],
            )

            # ImageNet normalization。
            image = (image - self.mean) / self.std

            return image, True, idx

        except Exception:
            # 失败图像不会进入 ConvNeXt，仅返回同尺寸占位张量便于 DataLoader 拼 batch。
            dummy = torch.zeros(
                (3, self.crop_size, self.crop_size),
                dtype=torch.float32,
            )
            return dummy, False, idx


def collate_fn(batch):
    """自定义 batch 拼接，同时保留读取成功标记和原始索引。"""
    images, flags, indices = zip(*batch)
    return (
        torch.stack(images),
        torch.tensor(flags, dtype=torch.bool),
        torch.tensor(indices, dtype=torch.long),
    )


# ============================================================================
# 2. ConvNeXt-Base 固定特征提取器
# ============================================================================

def create_feature_extractor(weights):
    """
    创建 torchvision 预训练 ConvNeXt-Base，并显式冻结全部参数。

    特征定义：
        ConvNeXt features -> avgpool -> flatten
    不进入最终 classifier，因此每张图像得到 1024 维特征。
    """
    base = models.convnext_base(weights=weights)

    # 冻结全部模型参数，仅提取图像特征。
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    # torchvision ConvNeXt 中 classifier 之前的输出通道数为 1024。
    extractor = nn.Sequential(
        base.features,
        base.avgpool,
    )
    extractor.eval()

    feature_dim = 1024
    return extractor, feature_dim


# ============================================================================
# 3. 从 preprocess 缓存整理图像级与 sequence 级元数据
# ============================================================================

def _safe_float(value, default=-1.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def flatten_examination_sequences(
    examination: Dict,
    class_to_idx: Dict[str, int],
) -> Tuple[Dict, List[Dict]]:
    """
    将一个 examination 的 acquisition groups 展平为图像列表，
    同时保留每张图像属于哪个 acquisition group。

    返回：
    1. image_meta：与每张图像一一对应的字段；
    2. sequence_meta：每个 acquisition group 的元数据。
    """
    image_meta = {
        "paths": [],
        "labels": [],
        "timestamps": [],
        "acquisition_seconds": [],
        "sequence_ids": [],
        "series_uids": [],
    }
    sequence_meta = []

    sequences = examination.get("sequences", [])

    for sequence_id, sequence in enumerate(sequences):
        paths = list(sequence.get("paths", []))
        if not paths:
            continue

        label_name = str(sequence.get("label", "")).lower()
        if label_name not in class_to_idx:
            raise ValueError(
                f"发现不在 class_names 中的标签: {label_name}; "
                f"合法类别={list(class_to_idx.keys())}"
            )

        label_idx = int(class_to_idx[label_name])
        timestamp = _safe_float(sequence.get("acquisition_time"), -1.0)
        acquisition_seconds = _safe_float(
            sequence.get("acquisition_seconds"),
            -1.0,
        )
        series_uid = str(sequence.get("series_uid", "") or "")

        image_meta["paths"].extend(paths)
        image_meta["labels"].extend([label_idx] * len(paths))
        image_meta["timestamps"].extend([timestamp] * len(paths))
        image_meta["acquisition_seconds"].extend(
            [acquisition_seconds] * len(paths)
        )
        image_meta["sequence_ids"].extend([sequence_id] * len(paths))
        image_meta["series_uids"].extend([series_uid] * len(paths))

        sequence_meta.append({
            "sequence_id": sequence_id,
            "label": label_idx,
            "label_name": label_name,
            "timestamp": timestamp,
            "acquisition_seconds": acquisition_seconds,
            "series_uid": series_uid,
            "original_image_count": len(paths),
        })

    return image_meta, sequence_meta


def filter_failed_images(
    features: np.ndarray,
    image_meta: Dict,
    sequence_meta: List[Dict],
    failed_indices: List[int],
) -> Tuple[np.ndarray, Dict, List[Dict]]:
    """
    删除加载失败的图像，并重新连续编号 sequence_ids。

    若某个 acquisition group 的所有图像都读取失败，则该 group 不写入有效
    sequence 级数据，避免后续构造 20 张输入时出现长度为 0 的 sequence。
    """
    n = len(image_meta["paths"])

    if failed_indices:
        valid_mask = np.ones(n, dtype=bool)
        valid_mask[np.asarray(failed_indices, dtype=np.int64)] = False
    else:
        valid_mask = np.ones(n, dtype=bool)

    filtered = {}
    for key, values in image_meta.items():
        arr = np.asarray(values, dtype=object)
        filtered[key] = arr[valid_mask].tolist()

    if len(features) != int(valid_mask.sum()):
        raise RuntimeError(
            "特征数量与成功读取图像数量不一致："
            f"features={len(features)}, valid_images={int(valid_mask.sum())}"
        )

    # 按第一次出现顺序重新连续编号 acquisition group。
    old_sequence_ids = filtered["sequence_ids"]
    ordered_old_ids = []
    seen = set()
    for sid in old_sequence_ids:
        sid = int(sid)
        if sid not in seen:
            seen.add(sid)
            ordered_old_ids.append(sid)

    old_to_new = {
        old_id: new_id
        for new_id, old_id in enumerate(ordered_old_ids)
    }
    filtered["sequence_ids"] = [
        old_to_new[int(sid)]
        for sid in old_sequence_ids
    ]

    sequence_meta_by_id = {
        int(item["sequence_id"]): item
        for item in sequence_meta
    }

    valid_sequence_meta = []
    sequence_id_array = np.asarray(filtered["sequence_ids"], dtype=np.int64)

    for old_id in ordered_old_ids:
        new_id = old_to_new[old_id]
        item = dict(sequence_meta_by_id[old_id])
        item["sequence_id"] = new_id
        item["valid_image_count"] = int((sequence_id_array == new_id).sum())
        valid_sequence_meta.append(item)

    return features, filtered, valid_sequence_meta


# ============================================================================
# 4. HDF5 写入辅助函数
# ============================================================================

def string_dtype():
    """HDF5 可变长度 UTF-8 字符串类型。"""
    return h5py.string_dtype(encoding="utf-8")


def write_examination_group(
    hf: h5py.File,
    exam_key: str,
    examination: Dict,
    features: np.ndarray,
    image_meta: Dict,
    sequence_meta: List[Dict],
):
    """将一个 examination 的图像特征和 sequence 边界写入 HDF5。"""
    safe_key = exam_key.replace("/", "_")
    group = hf.create_group(safe_key)

    group.create_dataset(
        "features",
        data=features.astype(np.float32, copy=False),
        compression="gzip",
        compression_opts=4,
    )
    group.create_dataset(
        "labels",
        data=np.asarray(image_meta["labels"], dtype=np.int64),
    )
    group.create_dataset(
        "timestamps",
        data=np.asarray(image_meta["timestamps"], dtype=np.float64),
    )
    group.create_dataset(
        "acquisition_seconds",
        data=np.asarray(image_meta["acquisition_seconds"], dtype=np.float64),
    )
    group.create_dataset(
        "sequence_ids",
        data=np.asarray(image_meta["sequence_ids"], dtype=np.int64),
    )
    group.create_dataset(
        "series_uids",
        data=np.asarray(image_meta["series_uids"], dtype=string_dtype()),
    )
    group.create_dataset(
        "paths",
        data=np.asarray(image_meta["paths"], dtype=string_dtype()),
    )

    # sequence 级信息，后续 Transformer 直接使用这些边界构造固定 20 张输入。
    group.create_dataset(
        "sequence_labels",
        data=np.asarray([x["label"] for x in sequence_meta], dtype=np.int64),
    )
    group.create_dataset(
        "sequence_timestamps",
        data=np.asarray([x["timestamp"] for x in sequence_meta], dtype=np.float64),
    )
    group.create_dataset(
        "sequence_seconds",
        data=np.asarray(
            [x["acquisition_seconds"] for x in sequence_meta],
            dtype=np.float64,
        ),
    )
    group.create_dataset(
        "sequence_lengths",
        data=np.asarray(
            [x["valid_image_count"] for x in sequence_meta],
            dtype=np.int64,
        ),
    )
    group.create_dataset(
        "sequence_series_uids",
        data=np.asarray(
            [x["series_uid"] for x in sequence_meta],
            dtype=string_dtype(),
        ),
    )

    # examination / participant 标识作为属性保存。
    group.attrs["participant_id"] = str(examination.get("participant_id", ""))
    group.attrs["exam_id"] = str(examination.get("exam_id", ""))
    group.attrs["study_uid"] = str(examination.get("study_uid", ""))
    group.attrs["study_date"] = str(examination.get("study_date", ""))
    group.attrs["sex"] = str(examination.get("sex", "U"))


# ============================================================================
# 5. 主特征提取流程
# ============================================================================

def extract_features(
    cache_prefix: str,
    output_dir: str,
    target_sex: str = "all",
    batch_size: int = 32,
    num_workers: int = 4,
    device_id: int = 0,
    use_amp: bool = False,
):
    """读取 preprocess 缓存，提取并保存 frozen ConvNeXt-Base 1024 维特征。"""
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(
        f"cuda:{device_id}"
        if torch.cuda.is_available()
        else "cpu"
    )

    preprocess_cfg = get_pretrained_preprocess_config()

    print("=" * 72)
    print("ConvNeXt-Base 固定特征提取")
    print(f"设备: {device}")
    print(f"权重: {preprocess_cfg['weights_name']}")
    print(
        "图像预处理: "
        "per-image min-max -> 0-255 -> 0-1 -> "
        f"resize(short side={preprocess_cfg['resize_size']}, "
        f"{preprocess_cfg['interpolation_name']}) "
        f"-> center crop({preprocess_cfg['crop_size']}x{preprocess_cfg['crop_size']}) "
        "-> 3通道 -> ImageNet normalization"
    )
    print(f"batch_size={batch_size}, num_workers={num_workers}, AMP={use_amp}")
    print(f"sex={target_sex}")
    print("=" * 72)

    if target_sex == "all":
        tags_to_process = ["male", "female", "unknown"]
    else:
        tags_to_process = [target_sex]

    for sex_tag in tags_to_process:
        cache_file = f"{cache_prefix}_{sex_tag}.pkl"

        if not os.path.exists(cache_file):
            print(f"\n跳过 {sex_tag}: 未找到 {cache_file}")
            continue

        print(f"\n{'=' * 72}")
        print(f"处理性别: {sex_tag}")
        print(f"输入缓存: {cache_file}")

        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)

        examination_data = cache_data["patient_data"]
        class_names = list(cache_data["class_names"])
        class_to_idx = {
            str(name).lower(): idx
            for idx, name in enumerate(class_names)
        }

        print(f"检查数: {len(examination_data)}")
        print(f"类别: {class_names}")

        # 创建一次模型，整个性别数据集复用。
        extractor, feature_dim = create_feature_extractor(
            preprocess_cfg["weights"]
        )
        extractor = extractor.to(device).eval()

        h5_path = os.path.join(
            output_dir,
            f"features_{sex_tag}.h5",
        )
        metadata_path = os.path.join(
            output_dir,
            f"metadata_{sex_tag}.pkl",
        )

        examination_keys = []
        participant_ids_by_exam = {}
        total_images = 0
        total_sequences = 0
        failed_count = 0
        skipped_examinations = 0

        with h5py.File(h5_path, "w") as hf:
            # 文件级元数据，便于以后核对实验参数。
            hf.attrs["feature_dim"] = feature_dim
            hf.attrs["weights_name"] = preprocess_cfg["weights_name"]
            hf.attrs["resize_size"] = preprocess_cfg["resize_size"]
            hf.attrs["crop_size"] = preprocess_cfg["crop_size"]
            hf.attrs["resize_rule"] = preprocess_cfg["resize_rule"]
            hf.attrs["mean"] = np.asarray(preprocess_cfg["mean"], dtype=np.float32)
            hf.attrs["std"] = np.asarray(preprocess_cfg["std"], dtype=np.float32)
            hf.attrs["feature_schema_version"] = METHODS_FEATURE_SCHEMA_VERSION
            hf.attrs["interpolation"] = preprocess_cfg["interpolation_name"]

            for exam_key in tqdm(
                examination_data.keys(),
                desc=f"提取 {sex_tag}",
            ):
                examination = examination_data[exam_key]

                image_meta, sequence_meta = flatten_examination_sequences(
                    examination,
                    class_to_idx,
                )

                all_paths = image_meta["paths"]
                if not all_paths:
                    skipped_examinations += 1
                    continue

                dataset = DicomDataset(
                    image_paths=all_paths,
                    resize_size=preprocess_cfg["resize_size"],
                    crop_size=preprocess_cfg["crop_size"],
                    mean=preprocess_cfg["mean"],
                    std=preprocess_cfg["std"],
                    interpolation=preprocess_cfg["interpolation"],
                )

                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                    pin_memory=(device.type == "cuda"),
                    persistent_workers=(num_workers > 0),
                )

                feature_chunks = []
                failed_indices = []

                # 默认使用 FP32；仅在显式启用时才使用混合精度。
                # 默认使用 FP32；只有显式 --amp 时才开启 CUDA autocast。
                if device.type == "cuda" and use_amp:
                    autocast_context = torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    )
                else:
                    autocast_context = nullcontext()

                with torch.inference_mode():
                    with autocast_context:
                        for images, flags, indices in loader:
                            batch_failed = indices[~flags].tolist()
                            failed_indices.extend(batch_failed)

                            valid_images = images[flags]
                            if valid_images.numel() == 0:
                                continue

                            valid_images = valid_images.to(
                                device,
                                non_blocking=True,
                            )

                            features = extractor(valid_images)
                            features = features.flatten(1)

                            if features.shape[1] != feature_dim:
                                raise RuntimeError(
                                    "ConvNeXt 特征维度异常："
                                    f"期望 {feature_dim}，实际 {features.shape[1]}"
                                )

                            feature_chunks.append(
                                features.float().cpu()
                            )

                if not feature_chunks:
                    skipped_examinations += 1
                    failed_count += len(failed_indices)
                    continue

                features = torch.cat(
                    feature_chunks,
                    dim=0,
                ).numpy()

                failed_count += len(failed_indices)

                features, filtered_meta, valid_sequence_meta = filter_failed_images(
                    features=features,
                    image_meta=image_meta,
                    sequence_meta=sequence_meta,
                    failed_indices=failed_indices,
                )

                if len(features) == 0 or not valid_sequence_meta:
                    skipped_examinations += 1
                    continue

                write_examination_group(
                    hf=hf,
                    exam_key=exam_key,
                    examination=examination,
                    features=features,
                    image_meta=filtered_meta,
                    sequence_meta=valid_sequence_meta,
                )

                examination_keys.append(exam_key)
                participant_ids_by_exam[exam_key] = str(
                    examination.get("participant_id", "")
                )
                total_images += len(features)
                total_sequences += len(valid_sequence_meta)

        # 继续保留 patient_ids 字段以兼容旧 evaluator，
        # 但它现在实际上保存的是 examination key。
        metadata = {
            "patient_ids": examination_keys,
            "examination_keys": examination_keys,
            "participant_ids_by_exam": participant_ids_by_exam,
            "feature_dim": feature_dim,
            "class_names": class_names,
            "source_cache": cache_file,
            "source_cache_schema_version": cache_data.get(
                "schema_version",
                "unknown",
            ),
            "feature_schema_version": METHODS_FEATURE_SCHEMA_VERSION,
            "weights_name": preprocess_cfg["weights_name"],
            "preprocessing": {
                "per_image_minmax": True,
                "intensity_rule": "per_image_minmax_then_scale_0_255_then_0_1",
                "resize_size": preprocess_cfg["resize_size"],
                "resize_rule": preprocess_cfg["resize_rule"],
                "center_crop_size": preprocess_cfg["crop_size"],
                "interpolation": preprocess_cfg["interpolation_name"],
                "three_channel_replication": True,
                "imagenet_mean": preprocess_cfg["mean"],
                "imagenet_std": preprocess_cfg["std"],
            },
            "sequence_boundary_source": (
                "preprocess_dicom_seq.py acquisition groups; stored as sequence_ids"
            ),
            "fixed_length_20_note": (
                "固定20张、zero padding和attention mask应在Transformer Dataset中实现，"
                "本特征脚本保存每个acquisition group的全部有效图像特征。"
            ),
        }

        with open(metadata_path, "wb") as f:
            pickle.dump(
                metadata,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        unique_participants = len(set(
            pid
            for pid in participant_ids_by_exam.values()
            if pid
        ))

        print(
            f"  完成 {sex_tag}: "
            f"{unique_participants} 个患者, "
            f"{len(examination_keys)} 次检查, "
            f"{total_sequences} 个 acquisition groups, "
            f"{total_images} 张有效图像, "
            f"读取失败 {failed_count} 张, "
            f"跳过检查 {skipped_examinations} 次"
        )
        print(f"  HDF5: {h5_path}")
        print(f"  元数据: {metadata_path}")

    print("\n全部特征提取完成。")


# ============================================================================
# 6. 命令行入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="按当前 Methods 提取 frozen ConvNeXt-Base 1024维特征"
    )
    parser.add_argument(
        "--cache_prefix",
        default="./patient_cache",
        help="输入缓存前缀，例如 ./patient_cache",
    )
    parser.add_argument(
        "--output_dir",
        default="./features_full",
        help="输出 HDF5 和 metadata 的目录",
    )
    parser.add_argument(
        "--sex_tag",
        choices=["all", "male", "female", "unknown"],
        default="all",
        help="指定要处理的性别",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="可选：GPU 上使用自动混合精度；默认关闭以提高复现一致性",
    )

    args = parser.parse_args()

    extract_features(
        cache_prefix=args.cache_prefix,
        output_dir=args.output_dir,
        target_sex=args.sex_tag,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_id=args.device_id,
        use_amp=args.amp,
    )
