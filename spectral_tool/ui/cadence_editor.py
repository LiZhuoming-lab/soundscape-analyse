from __future__ import annotations

import pandas as pd
import streamlit as st


def render_cadence_editor(cadence_annotations: pd.DataFrame, symbolic_analysis_key: str) -> pd.DataFrame:
    if cadence_annotations.empty:
        st.warning("当前规则下没有检测到终止候选。")
        return cadence_annotations

    st.caption(
        "当前终止检测分成两类：完满终止候选与半终止候选。"
        "完满终止更看重属准备、主到达、旋律骨架落在 1 与终止窗口的 5-1 支撑；"
        "半终止更看重属到达、属音骨架与句末收束。"
    )
    pac_candidates = cadence_annotations.loc[
        cadence_annotations["cadence_type"].astype(str) == "完满终止候选"
    ].copy()
    hc_candidates = cadence_annotations.loc[
        cadence_annotations["cadence_type"].astype(str) == "半终止候选"
    ].copy()
    tonic_skeleton_candidates = pac_candidates.loc[
        pac_candidates["melody_skeleton_class"].astype(str) == "主音骨架（1）"
    ].copy()
    summary_col_1, summary_col_2, summary_col_3 = st.columns(3)
    summary_col_1.metric("完满终止", int(len(pac_candidates)))
    summary_col_2.metric("半终止", int(len(hc_candidates)))
    summary_col_3.metric("主音骨架 PAC", int(len(tonic_skeleton_candidates)))

    if not tonic_skeleton_candidates.empty:
        st.markdown("**旋律骨架为 1 的候选**")
        st.dataframe(
            tonic_skeleton_candidates[
                [
                    "measure_number",
                    "cadence_window",
                    "cadence_key",
                    "candidate_score",
                    "melody_pitch_name",
                    "melody_scale_degree",
                    "melody_skeleton_class",
                    "previous_roman_numeral",
                    "current_roman_numeral",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    cadence_editor = st.data_editor(
        cadence_annotations[
            [
                "export",
                "cadence_type",
                "cadence_window",
                "cadence_key",
                "candidate_score",
                "dominant_score",
                "tonic_score",
                "melody_score",
                "bass_score",
                "closure_score",
                "measure_number",
                "beat",
                "melody_skeleton_class",
                "previous_roman_numeral",
                "current_roman_numeral",
                "previous_bass_scale_degree",
                "current_bass_scale_degree",
                "melody_pitch_name",
                "melody_scale_degree",
                "manual_label",
                "review_notes",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "export": st.column_config.CheckboxColumn("导出"),
            "cadence_type": st.column_config.TextColumn("系统判断", disabled=True),
            "cadence_window": st.column_config.TextColumn("终止窗口", disabled=True),
            "cadence_key": st.column_config.TextColumn("判定调性", disabled=True),
            "candidate_score": st.column_config.NumberColumn("总分", disabled=True, format="%.3f"),
            "dominant_score": st.column_config.NumberColumn("属准备", disabled=True, format="%.3f"),
            "tonic_score": st.column_config.NumberColumn("主到达", disabled=True, format="%.3f"),
            "melody_score": st.column_config.NumberColumn("旋律骨架", disabled=True, format="%.3f"),
            "bass_score": st.column_config.NumberColumn("低音支撑", disabled=True, format="%.3f"),
            "closure_score": st.column_config.NumberColumn("句末收束", disabled=True, format="%.3f"),
            "measure_number": st.column_config.NumberColumn("小节", disabled=True),
            "beat": st.column_config.NumberColumn("拍点", disabled=True),
            "melody_skeleton_class": st.column_config.TextColumn("旋律骨架分类", disabled=True),
            "previous_roman_numeral": st.column_config.TextColumn("前和声", disabled=True),
            "current_roman_numeral": st.column_config.TextColumn("当前和声", disabled=True),
            "previous_bass_scale_degree": st.column_config.NumberColumn("前窗口低音级数", disabled=True),
            "current_bass_scale_degree": st.column_config.NumberColumn("当前窗口低音级数", disabled=True),
            "melody_pitch_name": st.column_config.TextColumn("旋律音", disabled=True),
            "melody_scale_degree": st.column_config.NumberColumn("旋律音级", disabled=True),
            "manual_label": st.column_config.TextColumn("人工标签", width="medium"),
            "review_notes": st.column_config.TextColumn("备注", width="large"),
        },
        key=f"cadence_editor_{symbolic_analysis_key}",
    )
    cadence_annotations.loc[:, "export"] = cadence_editor["export"].astype(bool).to_numpy()
    cadence_annotations.loc[:, "manual_label"] = cadence_editor["manual_label"].astype(str).to_numpy()
    cadence_annotations.loc[:, "review_notes"] = cadence_editor["review_notes"].astype(str).to_numpy()
    return cadence_annotations

