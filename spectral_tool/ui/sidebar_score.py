from __future__ import annotations

import streamlit as st

from spectral_tool.symbolic_analysis import SymbolicAnalysisConfig


def render_score_sidebar() -> SymbolicAnalysisConfig:
    st.subheader("乐谱分析参数")
    theme_window_notes = st.slider("主题检索窗口（连续音数）", min_value=3, max_value=12, value=6, step=1)
    max_recurrence_results = st.slider("最多展示再现候选数", min_value=5, max_value=40, value=16, step=1)
    measure_summary_top_n = st.slider("每小节显示前几类音级 / 音高类", min_value=2, max_value=5, value=3, step=1)
    return SymbolicAnalysisConfig(
        theme_window_notes=theme_window_notes,
        max_recurrence_results=max_recurrence_results,
        measure_summary_top_n=measure_summary_top_n,
    )

