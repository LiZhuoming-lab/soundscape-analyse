from __future__ import annotations

import pandas as pd
import streamlit as st


def render_section_editor(annotations: pd.DataFrame, analysis_key: str) -> pd.DataFrame:
    if annotations.empty:
        return annotations

    st.markdown("**段落边界与标签列表**")
    editor_view = annotations[
        [
            "export",
            "section_id",
            "start_label",
            "end_label",
            "event_count",
            "auto_labels",
            "manual_labels",
            "review_notes",
            "descriptor",
        ]
    ].copy()

    edited = st.data_editor(
        editor_view,
        width="stretch",
        hide_index=True,
        column_config={
            "export": st.column_config.CheckboxColumn("导出"),
            "section_id": st.column_config.NumberColumn("段落", disabled=True),
            "start_label": st.column_config.TextColumn("起点", disabled=True),
            "end_label": st.column_config.TextColumn("终点", disabled=True),
            "event_count": st.column_config.NumberColumn("事件数", disabled=True),
            "auto_labels": st.column_config.TextColumn("自动标签", disabled=True, width="medium"),
            "manual_labels": st.column_config.TextColumn("人工标签", width="medium"),
            "review_notes": st.column_config.TextColumn("备注", width="large"),
            "descriptor": st.column_config.TextColumn("规则摘要", disabled=True, width="large"),
        },
        key=f"section_editor_{analysis_key}",
    )

    annotations.loc[:, "export"] = edited["export"].astype(bool).to_numpy()
    annotations.loc[:, "manual_labels"] = edited["manual_labels"].astype(str).to_numpy()
    annotations.loc[:, "review_notes"] = edited["review_notes"].astype(str).to_numpy()
    return annotations

