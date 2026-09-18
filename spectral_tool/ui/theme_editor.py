from __future__ import annotations

import pandas as pd
import streamlit as st


def render_theme_editor(theme_annotations: pd.DataFrame, symbolic_analysis_key: str) -> pd.DataFrame:
    if theme_annotations.empty:
        st.warning("当前窗口长度下没有检测到明显的主题 / 动机再现候选。")
        return theme_annotations

    theme_editor = st.data_editor(
        theme_annotations[
            [
                "export",
                "relation_type",
                "manual_relation",
                "similarity_score",
                "source_part",
                "source_measure",
                "source_excerpt",
                "match_part",
                "match_measure",
                "match_excerpt",
                "review_notes",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "export": st.column_config.CheckboxColumn("导出"),
            "relation_type": st.column_config.TextColumn("系统判断", disabled=True),
            "manual_relation": st.column_config.TextColumn("人工标签"),
            "similarity_score": st.column_config.NumberColumn("相似度", disabled=True),
            "source_part": st.column_config.TextColumn("源声部", disabled=True),
            "source_measure": st.column_config.NumberColumn("源小节", disabled=True),
            "source_excerpt": st.column_config.TextColumn("源片段", disabled=True, width="large"),
            "match_part": st.column_config.TextColumn("再现声部", disabled=True),
            "match_measure": st.column_config.NumberColumn("再现小节", disabled=True),
            "match_excerpt": st.column_config.TextColumn("再现片段", disabled=True, width="large"),
            "review_notes": st.column_config.TextColumn("备注", width="large"),
        },
        key=f"theme_editor_{symbolic_analysis_key}",
    )
    theme_annotations.loc[:, "export"] = theme_editor["export"].astype(bool).to_numpy()
    theme_annotations.loc[:, "manual_relation"] = theme_editor["manual_relation"].astype(str).to_numpy()
    theme_annotations.loc[:, "review_notes"] = theme_editor["review_notes"].astype(str).to_numpy()
    return theme_annotations
