from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st

from spectral_tool.symbolic_analysis import SymbolicAnalysisConfig


def build_symbolic_analysis_signature(score_bytes: bytes, config: SymbolicAnalysisConfig) -> str:
    payload = {
        "config": config.to_dict(),
        "score_sha1": hashlib.sha1(score_bytes).hexdigest(),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def ensure_harmony_annotation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ensured = frame.copy()
    if "manual_label" not in ensured.columns:
        default_labels = ensured.get("roman_numeral", pd.Series(dtype="object")).astype(str)
        fallback_labels = ensured.get("root", pd.Series(dtype="object")).astype(str)
        ensured["manual_label"] = default_labels.where(default_labels.str.strip() != "", fallback_labels)
    if "review_notes" not in ensured.columns:
        ensured["review_notes"] = ""
    if "export" not in ensured.columns:
        ensured["export"] = True
    return ensured


def ensure_theme_annotation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ensured = frame.copy()
    if "manual_relation" not in ensured.columns:
        ensured["manual_relation"] = ensured.get("relation_type", pd.Series(dtype="object")).astype(str)
    if "review_notes" not in ensured.columns:
        ensured["review_notes"] = ""
    if "export" not in ensured.columns:
        ensured["export"] = True
    return ensured


def ensure_cadence_annotation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ensured = frame.copy()
    if "melody_skeleton_class" not in ensured.columns:
        if "melody_scale_degree" in ensured.columns:
            ensured["melody_skeleton_class"] = ensured["melody_scale_degree"].apply(
                lambda value: "主音骨架（1）"
                if pd.notna(value) and int(value) == 1
                else f"非主音骨架（{int(value)}）"
                if pd.notna(value) and int(value) in {3, 5}
                else "骨架未定"
            )
        else:
            ensured["melody_skeleton_class"] = "骨架未定"
    if "manual_label" not in ensured.columns:
        ensured["manual_label"] = ensured.get("cadence_type", pd.Series(dtype="object")).astype(str)
    if "review_notes" not in ensured.columns:
        ensured["review_notes"] = ""
    if "export" not in ensured.columns:
        ensured["export"] = True
    return ensured


def sync_cadence_annotations(existing: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    latest_ensured = ensure_cadence_annotation_columns(latest)
    if existing.empty:
        return latest_ensured

    existing_ensured = ensure_cadence_annotation_columns(existing)
    identity_columns = ["cadence_type", "cadence_window", "cadence_key", "measure_number", "beat"]

    for column_name in identity_columns:
        if column_name not in existing_ensured.columns:
            existing_ensured[column_name] = None
        if column_name not in latest_ensured.columns:
            latest_ensured[column_name] = None

    existing_keyed = existing_ensured.set_index(identity_columns, drop=False)
    latest_keyed = latest_ensured.set_index(identity_columns, drop=False)

    editable_columns = ["manual_label", "review_notes", "export"]
    shared_keys = latest_keyed.index.intersection(existing_keyed.index)
    for column_name in editable_columns:
        if column_name not in existing_keyed.columns or column_name not in latest_keyed.columns:
            continue
        latest_keyed.loc[shared_keys, column_name] = existing_keyed.loc[shared_keys, column_name]

    return latest_keyed.reset_index(drop=True)


def cadence_result_signature(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "empty"
    normalized = frame.copy()
    ordered_columns = [
        "cadence_type",
        "cadence_window",
        "cadence_key",
        "measure_number",
        "beat",
        "candidate_score",
        "previous_roman_numeral",
        "current_roman_numeral",
    ]
    available_columns = [column_name for column_name in ordered_columns if column_name in normalized.columns]
    normalized = normalized[available_columns].sort_values(
        [column_name for column_name in ["measure_number", "beat", "cadence_type", "cadence_key"] if column_name in available_columns]
    ).reset_index(drop=True)
    return hashlib.sha1(normalized.to_json(orient="records", force_ascii=False).encode("utf-8")).hexdigest()


def init_harmony_annotations(result: dict[str, object], analysis_key: str) -> str:
    state_key = f"harmony_annotations_{analysis_key}"
    if state_key not in st.session_state:
        base = result["harmony_table"].copy()
        st.session_state[state_key] = ensure_harmony_annotation_columns(base)
    else:
        st.session_state[state_key] = ensure_harmony_annotation_columns(st.session_state[state_key])
    return state_key


def init_theme_annotations(result: dict[str, object], analysis_key: str) -> str:
    state_key = f"theme_annotations_{analysis_key}"
    if state_key not in st.session_state:
        base = result["theme_matches"].copy()
        st.session_state[state_key] = ensure_theme_annotation_columns(base)
    else:
        st.session_state[state_key] = ensure_theme_annotation_columns(st.session_state[state_key])
    return state_key


def init_cadence_annotations(result: dict[str, object], analysis_key: str) -> str:
    state_key = f"cadence_annotations_{analysis_key}"
    signature_key = f"{state_key}_result_signature"
    editor_key = f"cadence_editor_{analysis_key}"
    base = result["cadence_candidates"].copy()
    latest_signature = cadence_result_signature(base)
    if state_key not in st.session_state:
        st.session_state[state_key] = ensure_cadence_annotation_columns(base)
        st.session_state[signature_key] = latest_signature
    else:
        st.session_state[state_key] = sync_cadence_annotations(st.session_state[state_key], base)
        if st.session_state.get(signature_key) != latest_signature:
            st.session_state[signature_key] = latest_signature
            if editor_key in st.session_state:
                del st.session_state[editor_key]
    return state_key
