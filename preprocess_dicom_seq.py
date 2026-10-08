#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盆腔 DCE-MRI DICOM 数据预处理

处理步骤：
1. 扫描 DICOM 文件，提取患者、检查、序列及采集时间相关元数据；
2. 按 participant_id / StudyInstanceUID / SeriesInstanceUID 组织数据；
3. 同一序列内按采集时间和时间位置字段划分 acquisition group；
4. 根据序列名称筛选目标 T1 序列，并排除明确不属于盆腔的序列；
5. 分组后附加参考期相标签，整理组内图像顺序；
6. 按 PatientSex 输出 male / female / unknown 缓存及审计信息。

输出的每条检查记录保留 participant_id 和各 acquisition group，
供后续图像特征提取及患者级数据划分使用。
"""

import argparse
import json
import os
import pickle
import re
import warnings
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import pydicom
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ============================================================================
# 1. 基本配置
# ============================================================================

# 期相类别及其固定顺序。
CLASS_NAMES = ["precontrast", "arterial", "venous", "delayed"]  # precontrast: mask
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# 相邻图像的采集时间差超过 0.5 秒
# 或时间位置元数据发生变化时，开始新的 acquisition group。
ACQUISITION_GROUP_THRESHOLD_MILLISECONDS = 500
MILLISECONDS_PER_DAY = 24 * 60 * 60 * 1000

# 兼容旧数据目录中的类别名称。
LABEL_ALIASES = {
    "mask": "precontrast",
    "precontrast": "precontrast",
    "pre-contrast": "precontrast",
    "pre_contrast": "precontrast",
    "arterial": "arterial",
    "venous": "venous",
    "delay": "delayed",
    "delayed": "delayed",
}

# 保守的盆腔部位判断词。
# 只有明确属于其他部位的序列才排除；无法判断时保留并在审计结果中记录。
PELVIS_POSITIVE_TERMS = (
    "pelvis", "pelvic", "abdomen pelvis", "abdomen/pelvis", "abd pelvis",
    "盆腔", "骨盆",
)

PELVIS_NEGATIVE_TERMS = (
    "brain", "head", "neck", "cervical", "chest", "thorax", "lung",
    "cardiac", "heart", "shoulder", "elbow", "wrist", "hand", "knee",
    "ankle", "foot", "spine", "颅脑", "头部", "颈部", "胸部", "肺",
    "心脏", "肩", "肘", "腕", "膝", "踝", "足",
)


# ============================================================================
# 2. 通用辅助函数
# ============================================================================

def clean_str(value) -> str:
    """将 DICOM 字段安全转换为去除首尾空格后的字符串。"""
    if value is None:
        return ""
    return str(value).strip()


def parse_dicom_time(value) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """
    解析 DICOM AcquisitionTime。

    返回三个值：
    1. acquisition_time：完整 HHMMSS.frac 数值，保留小数秒；
    2. acquisition_seconds：从当天 00:00:00 开始的精确秒数，保留小数，用于排序；
    3. acquisition_second：只精确到“秒”的整数时间键，用于 acquisition group 分组。

    例如：
        10:25:59.842
    会得到：
        acquisition_time    = 102559.842
        acquisition_seconds = 37559.842
        acquisition_second  = 37559

    这样可以满足：
    - 分组时只看“秒”；
    - 排序时仍然使用真实 AcquisitionTime 的小数秒信息。
    """
    s = clean_str(value).replace(":", "")
    if not s:
        return None, None, None

    # DICOM TM 常见格式：HH、HHMM、HHMMSS、HHMMSS.frac
    m = re.match(r"^(\d{2})(\d{2})?(\d{2}(?:\.\d+)?)?", s)
    if not m:
        return None, None, None

    try:
        hh = int(m.group(1))
        mm = int(m.group(2) or 0)
        ss = float(m.group(3) or 0.0)

        if hh > 23 or mm > 59 or ss >= 60:
            return None, None, None

        # 保留与旧代码兼容的 HHMMSS.frac 数值形式。
        acquisition_time = hh * 10000 + mm * 100 + ss

        # 精确到小数秒，用于真实时间排序。
        acquisition_seconds = hh * 3600.0 + mm * 60.0 + ss

        # 只取整秒，保留用于兼容旧缓存和审计；正式分组使用毫秒阈值。
        acquisition_second = int(acquisition_seconds)

        return acquisition_time, acquisition_seconds, acquisition_second
    except Exception:
        return None, None, None


def canonical_label_from_path(path: str) -> Optional[str]:
    """
    从目录路径中寻找人工参考期相标签。

    注意：这里取得的标签只用于 acquisition group 构建完成后的参考标签赋值，
    不参与时间分组过程。
    """
    parts = [p.strip().lower() for p in os.path.normpath(path).split(os.sep)]
    for part in reversed(parts[:-1]):
        if part in LABEL_ALIASES:
            return LABEL_ALIASES[part]
    return None


def extract_series_folder_name(path: str) -> str:
    """
    尝试从目录结构中取得 series 目录名称。

    支持常见结构：
        data_dir/{patient}/{series}/{label}/*.dcm

    如果是：
        data_dir/{patient}/{label}/*.dcm
    则无法从目录可靠取得 series 名，后续会回退到 SeriesDescription / ProtocolName。
    """
    parts = os.path.normpath(path).split(os.sep)
    lower_parts = [p.lower() for p in parts]

    for i in range(len(parts) - 2, -1, -1):
        if lower_parts[i] in LABEL_ALIASES:
            if i - 1 >= 0:
                return parts[i - 1]
            break
    return ""


# ============================================================================
# 3. T1 序列识别
# ============================================================================

def normalize_series_name(series_name: str) -> str:
    """统一序列名称格式，便于统一使用关键词筛选规则。"""
    s = clean_str(series_name).lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def is_standard_t1_series(series_name: str) -> bool:
    """
    判断是否为目标 T1 序列。

    这段规则来自旧评估代码中保留的 is_standard_t1_series 逻辑：
    1. 序列名称必须以 T1 开头；
    2. 排除矢状位和冠状位；
    3. 排除 echo1 / echo2 / dualecho / Dixon / WFI 等衍生序列；
    4. 排除 water / fat / in-phase / opposed-phase；
    5. 排除 radial 序列。

    如果目录中没有可靠 series 名，再回退到 DICOM SeriesDescription / ProtocolName。
    """
    s = normalize_series_name(series_name)
    if not s:
        return False

    # 默认要求以 t1_ 开头，同时接受 SeriesDescription 恰好为 "T1" 的情况，
    # 这里同时接受恰好等于 "t1"。
    if not (s == "t1" or s.startswith("t1_")):
        return False

    # 排除非横断面。
    if "_sag_" in f"_{s}_" or "_cor_" in f"_{s}_":
        return False

    # 排除特殊重构或衍生序列。
    exclude_terms = (
        "echo1", "echo2", "dualecho", "dixon", "wfi",
        "_water", "_fat", "_ip", "_op", "radial",
    )
    if any(term in s for term in exclude_terms):
        return False

    return True


def series_is_target_t1(series_records: List[Dict]) -> bool:
    """
    判断一个 SeriesInstanceUID 是否属于目标 T1 序列。

    依次尝试：
    1. series 文件夹名；
    2. DICOM SeriesDescription；
    3. DICOM ProtocolName。

    只要其中一个名称满足目标 T1 规则，就保留该 Series。
    """
    candidates = []
    for r in series_records:
        candidates.extend([
            r.get("series_folder_name", ""),
            r.get("series_description", ""),
            r.get("protocol_name", ""),
        ])

    for name in candidates:
        if is_standard_t1_series(name):
            return True
    return False


# ============================================================================
# 4. 盆腔部位筛选
# ============================================================================

def classify_body_region(meta: Dict) -> str:
    """根据 DICOM 文本字段保守判断 pelvis / outside / unknown。"""
    text = " | ".join(
        clean_str(meta.get(k, "")).lower()
        for k in (
            "body_part_examined",
            "study_description",
            "series_description",
            "protocol_name",
        )
    )

    if any(term in text for term in PELVIS_POSITIVE_TERMS):
        return "pelvis"
    if any(term in text for term in PELVIS_NEGATIVE_TERMS):
        return "outside"
    return "unknown"


# ============================================================================
# 5. DICOM 元数据读取
# ============================================================================

def extract_dicom_metadata(path: str) -> Dict:
    """读取预处理所需的 DICOM 元数据，不加载像素数据。"""
    out = {
        "path": path,
        "read_ok": False,
        "reference_label": canonical_label_from_path(path),
        "series_folder_name": extract_series_folder_name(path),
        "participant_id": "",
        "patient_sex": "U",
        "study_date": "",
        "study_uid": "",
        "series_uid": "",
        "sop_uid": "",
        "acquisition_time": None,
        "acquisition_seconds": None,
        "acquisition_milliseconds": None,
        "acquisition_second": None,
        "instance_number": None,
        "temporal_position_identifier": None,
        "number_of_temporal_positions": None,
        "body_part_examined": "",
        "study_description": "",
        "series_description": "",
        "protocol_name": "",
        "scanning_sequence": "",
    }

    try:
        dcm = pydicom.dcmread(path, force=True, stop_before_pixels=True)
        out["read_ok"] = True

        out["participant_id"] = clean_str(getattr(dcm, "PatientID", ""))

        sex = clean_str(getattr(dcm, "PatientSex", "")).upper()
        out["patient_sex"] = sex if sex in {"M", "F"} else "U"

        out["study_date"] = clean_str(getattr(dcm, "StudyDate", ""))
        out["study_uid"] = clean_str(getattr(dcm, "StudyInstanceUID", ""))
        out["series_uid"] = clean_str(getattr(dcm, "SeriesInstanceUID", ""))
        out["sop_uid"] = clean_str(getattr(dcm, "SOPInstanceUID", ""))

        # 同时保存完整 AcquisitionTime 和只到秒的分组键。
        acq_time, acq_seconds, acq_second = parse_dicom_time(
            getattr(dcm, "AcquisitionTime", None)
        )
        out["acquisition_time"] = acq_time
        out["acquisition_seconds"] = acq_seconds
        if acq_seconds is not None:
            out["acquisition_milliseconds"] = int(round(acq_seconds * 1000.0))
        out["acquisition_second"] = acq_second

        try:
            value = getattr(dcm, "InstanceNumber", None)
            if value is not None and str(value).strip():
                out["instance_number"] = int(value)
        except Exception:
            pass

        try:
            value = getattr(dcm, "TemporalPositionIdentifier", None)
            if value is not None and str(value).strip():
                out["temporal_position_identifier"] = int(value)
        except Exception:
            pass

        try:
            value = getattr(dcm, "NumberOfTemporalPositions", None)
            if value is not None and str(value).strip():
                out["number_of_temporal_positions"] = int(value)
        except Exception:
            pass

        out["body_part_examined"] = clean_str(getattr(dcm, "BodyPartExamined", ""))
        out["study_description"] = clean_str(getattr(dcm, "StudyDescription", ""))
        out["series_description"] = clean_str(getattr(dcm, "SeriesDescription", ""))
        out["protocol_name"] = clean_str(getattr(dcm, "ProtocolName", ""))
        out["scanning_sequence"] = clean_str(getattr(dcm, "ScanningSequence", ""))
        out["body_region_status"] = classify_body_region(out)

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"

    return out


def mp_process(path: str) -> Dict:
    """多进程 worker。"""
    return extract_dicom_metadata(path)


def collect_dcm_files(root: str) -> List[str]:
    """递归收集目录中的全部 .dcm 文件。"""
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".dcm"):
                files.append(os.path.join(dirpath, fn))
    return files


# ============================================================================
# 6. 患者、检查和 Series 标识
# ============================================================================

def fallback_participant_id(path: str) -> str:
    """PatientID 缺失时，从已脱敏目录结构中尽量恢复一个患者标识。"""
    parts = os.path.normpath(path).split(os.sep)
    for part in reversed(parts[:-1]):
        if part.lower() in LABEL_ALIASES:
            continue
        if re.search(r"\d", part):
            return part
    return ""


def exam_id_for_record(record: Dict) -> str:
    """优先使用 StudyInstanceUID 区分同一患者的不同检查。"""
    if record.get("study_uid"):
        return record["study_uid"]
    if record.get("study_date"):
        return record["study_date"]
    return "unknown_exam"


def series_id_for_record(record: Dict) -> str:
    """优先使用 SeriesInstanceUID；缺失时使用所在目录作为稳定回退标识。"""
    if record.get("series_uid"):
        return record["series_uid"]
    return f"PATH::{os.path.dirname(record['path'])}"


# ============================================================================
# 7. 图像排序与 acquisition group 分组
# ============================================================================

def record_sort_key(record: Dict):
    """
    图像排序键。

    排序优先级：
    AcquisitionTime（毫秒）→ InstanceNumber → TemporalPositionIdentifier
    → NumberOfTemporalPositions。缺失的 AcquisitionTime、TemporalPositionIdentifier
    和 NumberOfTemporalPositions 使用 0，缺失的 InstanceNumber 使用 1；SOP UID
    和路径只作为最后的稳定性 tie-breaker。
    """
    acquisition_milliseconds = record.get("acquisition_milliseconds")
    if acquisition_milliseconds is None:
        acquisition_milliseconds = 0

    instance_number = record.get("instance_number")
    if instance_number is None:
        instance_number = 1

    temporal_position_identifier = record.get("temporal_position_identifier")
    if temporal_position_identifier is None:
        temporal_position_identifier = 0

    number_of_temporal_positions = record.get("number_of_temporal_positions")
    if number_of_temporal_positions is None:
        number_of_temporal_positions = 0

    return (
        acquisition_milliseconds,
        instance_number,
        temporal_position_identifier,
        number_of_temporal_positions,
        record.get("sop_uid") or "",
        record.get("path") or "",
    )


def split_series_into_acquisition_groups(series_records: List[Dict]) -> List[List[Dict]]:
    """
    将同一个 SeriesInstanceUID 拆分成 acquisition groups。

    分组过程：先按采集时间和时间位置元数据排序，
    然后按相邻图像的 AcquisitionTime 差值和时间位置元数据划分。相邻采集时间
    差超过 0.5 秒、TemporalPositionIdentifier 变化或
    NumberOfTemporalPositions 变化时，开始新的 group。单个序列可以因此包含多个
    phase group；参考标签在 group 构建完成后再赋值。
    """
    if not series_records:
        return []

    ordered = sorted(series_records, key=record_sort_key)
    groups: List[List[Dict]] = []
    current: List[Dict] = []
    previous: Optional[Dict] = None

    for record in ordered:
        if previous is None:
            current = [record]
            previous = record
            continue

        boundary = False

        previous_ms = previous.get("acquisition_milliseconds")
        current_ms = record.get("acquisition_milliseconds")
        if previous_ms is not None and current_ms is not None:
            # AcquisitionTime is a time-of-day field. The small wrap-around guard
            # prevents a scan crossing midnight from producing a huge negative gap.
            delta_ms = int(current_ms) - int(previous_ms)
            if delta_ms < 0:
                delta_ms += MILLISECONDS_PER_DAY
            boundary = delta_ms > ACQUISITION_GROUP_THRESHOLD_MILLISECONDS

        previous_tpi = previous.get("temporal_position_identifier")
        current_tpi = record.get("temporal_position_identifier")
        if (
            not boundary
            and previous_tpi is not None
            and current_tpi is not None
            and int(previous_tpi) != int(current_tpi)
        ):
            boundary = True

        previous_ntp = previous.get("number_of_temporal_positions")
        current_ntp = record.get("number_of_temporal_positions")
        if (
            not boundary
            and previous_ntp is not None
            and current_ntp is not None
            and int(previous_ntp) != int(current_ntp)
        ):
            boundary = True

        if boundary:
            groups.append(current)
            current = []

        current.append(record)
        previous = record

    if current:
        groups.append(current)

    return groups


# ============================================================================
# 8. acquisition group 参考标签和输出结构
# ============================================================================

def choose_group_label(
    group: List[Dict],
    mixed_label_policy: str,
) -> Tuple[Optional[str], bool]:
    """
    acquisition group 构建完成后，再确定该组的人工参考标签。

    如果一个时间组中出现多个不同人工标签，说明时间分组与人工标注发生冲突。
    默认跳过并记录，避免使用多数投票悄悄掩盖数据问题。
    """
    labels = [
        r.get("reference_label")
        for r in group
        if r.get("reference_label") in CLASS_TO_IDX
    ]

    if not labels:
        return None, False

    counts = Counter(labels)

    if len(counts) == 1:
        return labels[0], False

    if mixed_label_policy == "majority":
        return counts.most_common(1)[0][0], True

    if mixed_label_policy == "error":
        raise ValueError(
            f"同一个 acquisition group 中出现多个参考标签: {dict(counts)}"
        )

    return None, True


def summarize_group(
    group: List[Dict],
    label: str,
    series_uid: str,
    group_index: int,
) -> Dict:
    """将一个 acquisition group 整理成后续特征提取需要的结构。"""
    ordered = sorted(group, key=record_sort_key)

    acquisition_times = [
        r["acquisition_time"]
        for r in ordered
        if r.get("acquisition_time") is not None
    ]
    acquisition_seconds = [
        r["acquisition_seconds"]
        for r in ordered
        if r.get("acquisition_seconds") is not None
    ]
    acquisition_milliseconds = [
        r["acquisition_milliseconds"]
        for r in ordered
        if r.get("acquisition_milliseconds") is not None
    ]
    acquisition_second_keys = [
        r["acquisition_second"]
        for r in ordered
        if r.get("acquisition_second") is not None
    ]
    tpis = [
        r["temporal_position_identifier"]
        for r in ordered
        if r.get("temporal_position_identifier") is not None
    ]
    ntps = [
        r["number_of_temporal_positions"]
        for r in ordered
        if r.get("number_of_temporal_positions") is not None
    ]

    return {
        "group_index_within_series": group_index,
        "series_uid": series_uid,

        # 为兼容旧 feature_extractor / evaluator，acquisition_time 继续保存 HHMMSS.frac。
        "acquisition_time": min(acquisition_times) if acquisition_times else None,

        # 精确秒用于后续可靠的时间运算和排序。
        "acquisition_seconds": min(acquisition_seconds) if acquisition_seconds else None,
        "acquisition_seconds_end": max(acquisition_seconds) if acquisition_seconds else None,

        # 保存毫秒级时间，便于检查分组边界。
        "acquisition_milliseconds": (
            min(acquisition_milliseconds) if acquisition_milliseconds else None
        ),
        "acquisition_milliseconds_end": (
            max(acquisition_milliseconds) if acquisition_milliseconds else None
        ),

        # 记录该组实际对应的整秒键，便于兼容旧缓存和审计。
        "acquisition_second": (
            min(acquisition_second_keys) if acquisition_second_keys else None
        ),

        "temporal_position_identifier": min(tpis) if tpis else None,
        "number_of_temporal_positions": max(ntps) if ntps else None,
        "label": label,
        "label_idx": CLASS_TO_IDX[label],
        "paths": [r["path"] for r in ordered],
        "instance_numbers": [r.get("instance_number") for r in ordered],
        "series_folder_name": ordered[0].get("series_folder_name", "") if ordered else "",
        "series_description": ordered[0].get("series_description", "") if ordered else "",
        "protocol_name": ordered[0].get("protocol_name", "") if ordered else "",
        "body_region_status": ordered[0].get("body_region_status", "unknown") if ordered else "unknown",
    }


# ============================================================================
# 9. 构建 examination 级数据
# ============================================================================

def build_examinations(
    records: List[Dict],
    body_region_filter: bool,
    t1_filter: bool,
    mixed_label_policy: str,
    require_precontrast_and_enhanced: bool,
) -> Tuple[Dict, Dict]:
    """从 DICOM 元数据构建 examination → series → acquisition group 层级。"""
    audit = Counter()
    exams_raw = defaultdict(list)

    # ----------------------------------------------------------------------
    # 第一步：先按患者和检查组织所有可读 DICOM。
    # ----------------------------------------------------------------------
    for record in records:
        if not record.get("read_ok"):
            audit["corrupt_or_unreadable_files"] += 1
            continue

        if record.get("reference_label") not in CLASS_TO_IDX:
            audit["files_without_supported_reference_label"] += 1
            continue

        if not record.get("participant_id"):
            record["participant_id"] = fallback_participant_id(record["path"])

        if not record.get("participant_id"):
            audit["files_missing_participant_id"] += 1
            continue

        exam_id = exam_id_for_record(record)
        exam_key = f"{record['participant_id']}__{exam_id}"
        exams_raw[exam_key].append(record)

    exam_data = {}

    # ----------------------------------------------------------------------
    # 第二步：逐个 examination 处理其中的 SeriesInstanceUID。
    # ----------------------------------------------------------------------
    for exam_key, exam_records in exams_raw.items():
        participant_id = exam_records[0]["participant_id"]
        exam_id = exam_id_for_record(exam_records[0])
        study_uid = exam_records[0].get("study_uid", "")
        study_date = exam_records[0].get("study_date", "")

        by_series = defaultdict(list)
        for record in exam_records:
            by_series[series_id_for_record(record)].append(record)

        sequences = []
        sex_votes = []

        for series_uid, series_records in by_series.items():
            # --------------------------------------------------------------
            # T1 筛选：按序列名称关键词进行筛选。
            # --------------------------------------------------------------
            if t1_filter and not series_is_target_t1(series_records):
                audit["series_excluded_non_t1"] += 1
                audit["files_in_excluded_non_t1_series"] += len(series_records)
                continue

            # --------------------------------------------------------------
            # 盆腔部位筛选。
            # --------------------------------------------------------------
            region_states = [
                r.get("body_region_status", "unknown")
                for r in series_records
            ]

            if "pelvis" in region_states:
                series_region = "pelvis"
            elif region_states and all(x == "outside" for x in region_states):
                series_region = "outside"
            else:
                series_region = "unknown"

            if body_region_filter and series_region == "outside":
                audit["series_excluded_outside_pelvis"] += 1
                audit["files_in_excluded_outside_pelvis_series"] += len(series_records)
                continue

            if series_region == "unknown":
                audit["series_body_region_unknown_retained"] += 1

            # --------------------------------------------------------------
            # 缺少 AcquisitionTime 和 InstanceNumber 的序列无法可靠排序，予以排除。
            # --------------------------------------------------------------
            has_acquisition_time = any(
                r.get("acquisition_second") is not None
                for r in series_records
            )
            has_instance_number = any(
                r.get("instance_number") is not None
                for r in series_records
            )

            if not has_acquisition_time and not has_instance_number:
                audit["series_excluded_missing_acqtime_and_instance"] += 1
                audit["files_in_excluded_metadata_poor_series"] += len(series_records)
                continue

            # --------------------------------------------------------------
            # 同一个 SeriesInstanceUID 内按 0.5 秒阈值和时间位置元数据分组。
            # --------------------------------------------------------------
            groups = split_series_into_acquisition_groups(series_records)
            audit["acquisition_groups_constructed"] += len(groups)

            for group_index, group in enumerate(groups):
                label, mixed = choose_group_label(group, mixed_label_policy)

                if mixed:
                    audit["mixed_reference_label_groups"] += 1

                if label is None:
                    audit["acquisition_groups_skipped_no_unique_reference_label"] += 1
                    continue

                seq = summarize_group(
                    group=group,
                    label=label,
                    series_uid=series_uid,
                    group_index=group_index,
                )
                seq["body_region_status"] = series_region
                sequences.append(seq)

                sex_votes.extend([
                    r.get("patient_sex", "U")
                    for r in group
                ])

        if not sequences:
            audit["examinations_without_usable_groups"] += 1
            continue

        # ------------------------------------------------------------------
        # examination 内各 acquisition group 按真实时间和时间位置元数据排序。
        # ------------------------------------------------------------------
        sequences.sort(
            key=lambda s: (
                s.get("acquisition_milliseconds")
                if s.get("acquisition_milliseconds") is not None else 0,
                min(
                    [x for x in s.get("instance_numbers", []) if x is not None],
                    default=1,
                ),
                s.get("temporal_position_identifier")
                if s.get("temporal_position_identifier") is not None else 0,
                s.get("number_of_temporal_positions")
                if s.get("number_of_temporal_positions") is not None else 0,
                s.get("series_uid", ""),
            )
        )

        # ------------------------------------------------------------------
        # 检查至少应包含平扫以及一种增强期相。
        # ------------------------------------------------------------------
        phase_set = {s["label"] for s in sequences}
        if require_precontrast_and_enhanced:
            has_precontrast = "precontrast" in phase_set
            has_enhanced = any(
                phase in phase_set
                for phase in ("arterial", "venous", "delayed")
            )

            if not (has_precontrast and has_enhanced):
                audit["examinations_excluded_missing_precontrast_or_enhanced"] += 1
                continue

        sex_counter = Counter(
            x for x in sex_votes
            if x in {"M", "F"}
        )
        sex = sex_counter.most_common(1)[0][0] if sex_counter else "U"

        exam_data[exam_key] = {
            "participant_id": participant_id,
            "exam_id": exam_id,
            "study_uid": study_uid,
            "study_date": study_date,
            "sex": sex,
            "sequences": sequences,
        }

        audit["examinations_retained"] += 1
        audit["acquisition_groups_retained"] += len(sequences)
        audit["images_retained"] += sum(
            len(seq["paths"])
            for seq in sequences
        )

    return exam_data, dict(audit)


# ============================================================================
# 10. 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="盆腔 DCE-MRI DICOM 预处理"
    )

    parser.add_argument(
        "--data_dir",
        required=True,
        help="包含已标注 DICOM 文件的根目录",
    )
    parser.add_argument(
        "--output_prefix",
        default="./patient_cache",
        help="输出文件前缀",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="并行读取 DICOM 元数据的进程数",
    )
    parser.add_argument(
        "--mixed_label_policy",
        choices=["skip", "majority", "error"],
        default="skip",
        help="同一 acquisition group 出现多个参考标签时的处理方式",
    )
    parser.add_argument(
        "--disable_t1_filter",
        action="store_true",
        help="关闭旧代码中的标准 T1 序列筛选，仅用于调试",
    )
    parser.add_argument(
        "--disable_body_region_filter",
        action="store_true",
        help="关闭明确非盆腔序列的过滤，仅用于调试",
    )
    parser.add_argument(
        "--allow_incomplete_exam",
        action="store_true",
        help="允许缺少平扫或增强期的检查进入缓存",
    )

    # 保留旧命令行参数，避免已有 shell 脚本直接报错。
    # 期相标签取自分组后的参考标注，不按固定时间间隔自动重标。
    parser.add_argument(
        "--disable_auto_delay",
        action="store_true",
        help="兼容旧脚本的废弃参数，当前版本无实际作用",
    )

    args = parser.parse_args()

    os.makedirs(
        os.path.dirname(args.output_prefix) or ".",
        exist_ok=True,
    )

    print("=" * 72)
    print("盆腔 DCE-MRI DICOM 预处理")
    print(f"数据目录: {args.data_dir}")
    print(
        "AcquisitionTime 分组方式: 相邻时间差 > 0.5 秒，"
        "并结合 TemporalPositionIdentifier/NumberOfTemporalPositions"
    )
    print(
        "排序方式: AcquisitionTime(毫秒) -> InstanceNumber -> "
        "TemporalPositionIdentifier -> NumberOfTemporalPositions"
    )
    print(f"T1 筛选: {'关闭' if args.disable_t1_filter else '开启'}")
    print(
        f"盆腔部位筛选: "
        f"{'关闭' if args.disable_body_region_filter else '开启（保守）'}"
    )
    print(f"混合标签处理方式: {args.mixed_label_policy}")
    print("=" * 72)

    # ----------------------------------------------------------------------
    # 第 1 步：收集 DICOM 文件。
    # ----------------------------------------------------------------------
    print("\n[1/5] 收集 DICOM 文件...")
    dcm_files = collect_dcm_files(args.data_dir)
    print(f"  找到 {len(dcm_files)} 个 .dcm 文件")

    if not dcm_files:
        raise ValueError(
            f"在目录中没有找到 .dcm 文件: {args.data_dir}"
        )

    # ----------------------------------------------------------------------
    # 第 2 步：并行读取 DICOM 元数据。
    # ----------------------------------------------------------------------
    workers = max(
        1,
        min(args.num_workers or 1, len(dcm_files), cpu_count()),
    )

    print(f"\n[2/5] 使用 {workers} 个进程读取 DICOM 元数据...")

    with Pool(workers) as pool:
        records = list(
            tqdm(
                pool.imap_unordered(
                    mp_process,
                    dcm_files,
                    chunksize=200,
                ),
                total=len(dcm_files),
                desc="  读取 DICOM 元数据",
            )
        )

    readable = sum(
        1 for record in records
        if record.get("read_ok")
    )
    print(f"  可读取文件: {readable}/{len(records)}")

    # ----------------------------------------------------------------------
    # 第 3 步：构建 examination / series / acquisition group。
    # ----------------------------------------------------------------------
    print("\n[3/5] 构建检查、Series 和 acquisition group...")

    exam_data, audit = build_examinations(
        records=records,
        body_region_filter=not args.disable_body_region_filter,
        t1_filter=not args.disable_t1_filter,
        mixed_label_policy=args.mixed_label_policy,
        require_precontrast_and_enhanced=not args.allow_incomplete_exam,
    )

    participant_ids = {
        exam["participant_id"]
        for exam in exam_data.values()
    }

    print(f"  保留患者数: {len(participant_ids)}")
    print(f"  保留检查数: {len(exam_data)}")
    print(
        "  保留 acquisition group 数: "
        f"{sum(len(exam['sequences']) for exam in exam_data.values())}"
    )

    # ----------------------------------------------------------------------
    # 第 4 步：按照 PatientSex 分别保存。
    # ----------------------------------------------------------------------
    print("\n[4/5] 按 DICOM PatientSex 拆分并保存...")

    sex_map = {
        "M": "male",
        "F": "female",
    }
    outputs = defaultdict(dict)

    for exam_key, exam in exam_data.items():
        sex_tag = sex_map.get(
            exam.get("sex", "U"),
            "unknown",
        )
        outputs[sex_tag][exam_key] = exam

    for sex_tag in ("male", "female", "unknown"):
        data = outputs.get(sex_tag, {})
        if not data:
            continue

        output_path = f"{args.output_prefix}_{sex_tag}.pkl"

        with open(output_path, "wb") as f:
            pickle.dump(
                {
                    # 为兼容旧 feature_extractor，继续保留 patient_data 这个顶层键名。
                    # 但其中每一条记录现在是 examination 级，真正的患者 ID 存在 participant_id。
                    "patient_data": data,
                    "class_names": CLASS_NAMES,
                    "schema_version": "methods_2026_v3_0p5s_temporal_grouping",
                    "grouping_rule": (
                        "within each SeriesInstanceUID, split when adjacent "
                        "AcquisitionTime gap exceeds 500 ms or temporal-position "
                        "metadata changes"
                    ),
                    "note": (
                        "做患者级训练/验证/测试划分时必须使用 participant_id，"
                        "不要直接使用 patient_data 的 examination key。"
                    ),
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        n_participants = len({
            exam["participant_id"]
            for exam in data.values()
        })
        n_groups = sum(
            len(exam["sequences"])
            for exam in data.values()
        )
        n_images = sum(
            sum(len(seq["paths"]) for seq in exam["sequences"])
            for exam in data.values()
        )

        print(
            f"  ✓ {sex_tag}: "
            f"{n_participants} 个患者, "
            f"{len(data)} 次检查, "
            f"{n_groups} 个 acquisition groups, "
            f"{n_images} 张图像 -> {output_path}"
        )

    # ----------------------------------------------------------------------
    # 第 5 步：保存审计信息。
    # ----------------------------------------------------------------------
    print("\n[5/5] 保存预处理审计信息...")

    audit_payload = {
        "input_dicom_files": len(dcm_files),
        "readable_dicom_files": readable,
        "class_names": CLASS_NAMES,
        "grouping_rule": (
            "within each SeriesInstanceUID, split when adjacent AcquisitionTime "
            "gap exceeds 500 ms or TemporalPositionIdentifier/"
            "NumberOfTemporalPositions changes; sort by AcquisitionTime(ms), "
            "InstanceNumber, TemporalPositionIdentifier, NumberOfTemporalPositions"
        ),
        "t1_filter": (
            "disabled"
            if args.disable_t1_filter
            else "legacy is_standard_t1_series rule"
        ),
        "body_region_filter": (
            "disabled"
            if args.disable_body_region_filter
            else "conservative"
        ),
        "require_precontrast_and_enhanced": not args.allow_incomplete_exam,
        "mixed_label_policy": args.mixed_label_policy,
        "retained_participants": len(participant_ids),
        "retained_examinations": len(exam_data),
        "counts": audit,
        "notes": [
            "文件夹期相标签仅作为人工参考标签，不参与 acquisition group 构建。",
            "固定 20 张图像和 zero padding 留到后续 Transformer 输入构建阶段。",
            "T1 筛选逻辑复用了旧评估代码中的 is_standard_t1_series 规则。",
        ],
    }

    audit_path = f"{args.output_prefix}_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(
            audit_payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"  ✓ {audit_path}")
    print("\n预处理完成。")

    if outputs.get("unknown"):
        print(
            "警告：存在 PatientSex 未知的检查，已单独写入 unknown 缓存；"
            "最终男女分模型分析时不应直接纳入。"
        )


if __name__ == "__main__":
    main()
