from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import BinaryIO

import pandas as pd
import streamlit as st

from spectral_tool.analysis import (
    ANALYSIS_METHOD_VERSION,
    AnalysisConfig,
    format_seconds,
    split_label_text,
)


def _source_sha256(source: bytes | bytearray | str | Path | BinaryIO) -> str:
    digest = hashlib.sha256()
    if isinstance(source, (bytes, bytearray)):
        digest.update(source)
        return digest.hexdigest()
    if isinstance(source, (str, Path)):
        with Path(source).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    position = source.tell() if hasattr(source, "tell") else None
    try:
        source.seek(0)
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    finally:
        if position is not None:
            source.seek(position)
    return digest.hexdigest()


def build_analysis_signature(
    audio_source: bytes | bytearray | str | Path | BinaryIO,
    config: AnalysisConfig,
    channel_mode: str,
    audio_sha256: str | None = None,
) -> str:
    payload = {
        "method_version": ANALYSIS_METHOD_VERSION,
        "channel_mode": channel_mode,
        "config": config.to_dict(),
        "audio_sha256": audio_sha256 or _source_sha256(audio_source),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def source_sha256(source: bytes | bytearray | str | Path | BinaryIO) -> str:
    return _source_sha256(source)


def ensure_event_annotation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ensured = frame.copy()
    if "manual_labels" not in ensured.columns:
        if "auto_labels" in ensured.columns:
            ensured["manual_labels"] = ensured["auto_labels"].astype(str)
        else:
            ensured["manual_labels"] = pd.Series(dtype="object")
    if "review_notes" not in ensured.columns:
        ensured["review_notes"] = ""
    if "export" not in ensured.columns:
        ensured["export"] = True
    if "manual_time_sec" not in ensured.columns:
        ensured["manual_time_sec"] = ensured["time_sec"] if "time_sec" in ensured.columns else pd.Series(dtype="float64")
    if "manual_time_label" not in ensured.columns:
        if "time_label" in ensured.columns:
            ensured["manual_time_label"] = ensured["time_label"].astype(str)
        else:
            ensured["manual_time_label"] = ""
    if "operation_result" not in ensured.columns:
        ensured["operation_result"] = ""
    if "operation_log" not in ensured.columns:
        ensured["operation_log"] = ""
    return ensured


def init_event_annotations(result: dict[str, object], analysis_key: str) -> str:
    state_key = f"event_annotations_{analysis_key}"
    if state_key not in st.session_state:
        base = result["event_table"].copy()
        if base.empty:
            st.session_state[state_key] = base
        else:
            st.session_state[state_key] = ensure_event_annotation_columns(base)
    else:
        st.session_state[state_key] = ensure_event_annotation_columns(st.session_state[state_key])
    return state_key


def init_section_annotations(result: dict[str, object], analysis_key: str) -> str:
    state_key = f"section_annotations_{analysis_key}"
    if state_key not in st.session_state:
        base = result["section_table"].copy()
        if base.empty:
            st.session_state[state_key] = base
        else:
            base["manual_labels"] = base["auto_labels"]
            base["review_notes"] = ""
            base["export"] = True
            st.session_state[state_key] = base
    return state_key


def event_label(options: list[int], annotations: pd.DataFrame, event_id: int) -> str:
    row = annotations.loc[annotations["event_id"] == event_id].iloc[0]
    priority_label = str(row.get("interaction_priority_label", "")).strip()
    prefix = f"[{priority_label}] " if priority_label else ""
    display_time = str(row.get("manual_time_label", "")).strip() or str(row["time_label"])
    return f"{prefix}事件 {event_id} | {display_time} | {row['manual_labels'] or row['auto_labels']}"


def section_label(annotations: pd.DataFrame, section_id: int) -> str:
    row = annotations.loc[annotations["section_id"] == section_id].iloc[0]
    return f"段落 {section_id} | {row['start_label']} - {row['end_label']}"


def effective_event_labels(row: pd.Series) -> str:
    manual = str(row.get("manual_labels", "")).strip()
    auto = str(row.get("auto_labels", "")).strip()
    return manual or auto


def available_event_filter_labels(annotations: pd.DataFrame, preferred_vocab: list[str]) -> list[str]:
    observed: list[str] = []
    for _, row in annotations.iterrows():
        observed.extend(split_label_text(effective_event_labels(row)))

    ordered: list[str] = []
    seen: set[str] = set()
    for label in preferred_vocab + observed:
        if label and label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def filter_event_annotations(annotations: pd.DataFrame, selected_labels: list[str]) -> pd.DataFrame:
    if annotations.empty or not selected_labels:
        return annotations.copy()

    selected_set = set(selected_labels)
    mask = annotations.apply(
        lambda row: bool(selected_set.intersection(split_label_text(effective_event_labels(row)))),
        axis=1,
    )
    return annotations.loc[mask].copy()


def format_operation_log(
    mode_label: str,
    row: pd.Series,
    operation_result: str,
    manual_time_sec: float,
    review_notes: str,
) -> str:
    original_time_label = str(row.get("time_label", "--"))
    adjusted_time_label = format_seconds(float(manual_time_sec))
    pre_rms = max(float(row.get("pre_rms", 0.0)), 1e-8)
    post_rms = float(row.get("post_rms", 0.0))
    pre_centroid = max(float(row.get("pre_centroid_hz", 1.0)), 1.0)
    post_centroid = float(row.get("post_centroid_hz", 0.0))
    pre_flatness = float(row.get("pre_flatness", 0.0))
    post_flatness = float(row.get("post_flatness", 0.0))
    pre_high_ratio = float(row.get("pre_high_ratio", 0.0))
    post_high_ratio = float(row.get("post_high_ratio", 0.0))
    rms_ratio = post_rms / pre_rms
    centroid_ratio = post_centroid / pre_centroid
    flatness_delta = post_flatness - pre_flatness
    high_delta = post_high_ratio - pre_high_ratio
    lines = [
        f"事件 {int(row.get('event_id', 0))}",
        f"当前模式：{mode_label}",
        f"原始时间：{original_time_label}",
        f"人工时间：{adjusted_time_label}",
        f"当前自动标签：{str(row.get('auto_labels', ''))}",
        f"当前人工标签：{str(row.get('manual_labels', ''))}",
        (
            "当前事件特征："
            f"rms_ratio={rms_ratio:.3f}, "
            f"centroid_ratio={centroid_ratio:.3f}, "
            f"flatness_delta={flatness_delta:.3f}, "
            f"high_delta={high_delta:.3f}"
        ),
        f"你的操作结果：{operation_result or '未记录'}",
    ]
    note_text = review_notes.strip()
    if note_text:
        lines.append(f"说明：{note_text}")
    return "\n".join(lines)
