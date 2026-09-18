from __future__ import annotations

import pandas as pd
import streamlit as st


def render_harmony_editor(harmony_annotations: pd.DataFrame, symbolic_result: dict[str, object], symbolic_analysis_key: str) -> pd.DataFrame:
    if harmony_annotations.empty:
        st.warning("当前乐谱没有生成有效的和声切片。")
        return harmony_annotations

    harmony_editor = st.data_editor(
        harmony_annotations[
            [
                "export",
                "slice_id",
                "measure_number",
                "beat",
                "pitch_names",
                "roman_numeral",
                "manual_label",
                "review_notes",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "export": st.column_config.CheckboxColumn("导出"),
            "slice_id": st.column_config.NumberColumn("切片", disabled=True),
            "measure_number": st.column_config.NumberColumn("小节", disabled=True),
            "beat": st.column_config.NumberColumn("拍点", disabled=True),
            "pitch_names": st.column_config.TextColumn("垂直音响", disabled=True, width="large"),
            "roman_numeral": st.column_config.TextColumn("候选级数", disabled=True),
            "manual_label": st.column_config.TextColumn("人工标签", width="medium"),
            "review_notes": st.column_config.TextColumn("备注", width="large"),
        },
        key=f"harmony_editor_{symbolic_analysis_key}",
    )
    harmony_annotations.loc[:, "export"] = harmony_editor["export"].astype(bool).to_numpy()
    harmony_annotations.loc[:, "manual_label"] = harmony_editor["manual_label"].astype(str).to_numpy()
    harmony_annotations.loc[:, "review_notes"] = harmony_editor["review_notes"].astype(str).to_numpy()

    if not symbolic_result["harmony_table"].empty:
        harmony_counts = (
            symbolic_result["harmony_table"]["roman_numeral"]
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(12)
        )
        if not harmony_counts.empty:
            st.markdown("**最常见和声候选**")
            st.bar_chart(harmony_counts)

    return harmony_annotations

