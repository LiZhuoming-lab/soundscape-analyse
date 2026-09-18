from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from spectral_tool.analysis import build_audio_excerpt_wav, format_seconds, join_labels, split_label_text
from spectral_tool.assistant import (
    annotate_event_interaction_levels,
    annotate_event_similarity_groups,
)
from spectral_tool.models.presets import EVENT_MODEL_PRESETS, OPERATION_RESULT_OPTIONS
from spectral_tool.state.audio_state import (
    available_event_filter_labels,
    effective_event_labels,
    event_label,
    filter_event_annotations,
    format_operation_log,
)
from spectral_tool.visualization import (
    build_feature_curve_chart,
    build_interactive_novelty_chart,
    build_novelty_threshold_chart,
    build_synced_waveform_player_html,
    extract_selected_point_value,
    figure_to_png_bytes,
    plot_local_spectrogram,
)


def render_event_editor(
    annotations: pd.DataFrame,
    result: dict[str, object],
    analysis_key: str,
    preset_key: str,
) -> tuple[pd.DataFrame, int | None]:
    if annotations.empty:
        return annotations, None

    filter_key = f"event_label_filter_{analysis_key}"
    filter_options = available_event_filter_labels(annotations, list(result["event_label_vocab"]))
    selected_filter_labels = st.multiselect(
        "按标签筛选事件图",
        options=filter_options,
        key=filter_key,
        help="优先使用人工标签筛选；如果某个事件还没有人工标签，则回退使用自动标签。",
    )

    interaction_annotations = annotations.copy()
    interaction_annotations["effective_labels"] = interaction_annotations.apply(effective_event_labels, axis=1)
    interaction_annotations = annotate_event_interaction_levels(interaction_annotations)
    interaction_annotations = annotate_event_similarity_groups(interaction_annotations, threshold=0.90)
    filtered_annotations = filter_event_annotations(interaction_annotations, selected_filter_labels)
    if selected_filter_labels:
        st.caption(f"当前筛选后显示 {len(filtered_annotations)} / {len(annotations)} 个事件。")
    else:
        st.caption(f"当前显示全部 {len(annotations)} 个事件。")

    if filtered_annotations.empty:
        st.warning("当前标签筛选下没有匹配事件。可以清空筛选，或者换一个标签试试。")
        return annotations, None

    show_minor_key = f"show_minor_changes_{analysis_key}"
    show_minor_changes = st.toggle(
        "显示更多较小变化",
        value=False,
        key=show_minor_key,
        help="默认优先突出更值得互动讨论的事件；打开后可把较小变化也放进当前交互层里。",
    )
    weak_count = int((filtered_annotations["interaction_priority"] == "weak").sum())
    strong_count = int((filtered_annotations["interaction_priority"] == "strong").sum())
    medium_count = int((filtered_annotations["interaction_priority"] == "medium").sum())
    st.caption(f"当前交互层级分布：重点变化 {strong_count} 个，一般变化 {medium_count} 个，较小变化 {weak_count} 个。")

    visible_annotations = filtered_annotations.copy()
    if not show_minor_changes:
        visible_annotations = filtered_annotations.loc[filtered_annotations["interaction_priority"] != "weak"].copy()
        if visible_annotations.empty:
            visible_annotations = filtered_annotations.copy()
        elif weak_count:
            st.caption("较小变化默认收起；如果你也想把它们放进同一层互动，可以打开“显示更多较小变化”。")

    selector_key = f"active_event_{analysis_key}"
    event_ids = visible_annotations["event_id"].astype(int).tolist()
    if selector_key not in st.session_state or int(st.session_state[selector_key]) not in event_ids:
        st.session_state[selector_key] = event_ids[0]

    chart_source = visible_annotations.copy()
    chart_source["auto_labels"] = chart_source.apply(effective_event_labels, axis=1)
    st.altair_chart(
        build_novelty_threshold_chart(result["feature_table"]),
        width="stretch",
    )
    selected_from_chart = st.altair_chart(
        build_interactive_novelty_chart(result["feature_table"], chart_source),
        width="stretch",
        on_select="rerun",
        selection_mode="event_pick",
        key=f"novelty_pick_{analysis_key}",
    )
    clicked_event = extract_selected_point_value(selected_from_chart, "event_pick", "event_id")
    if clicked_event is not None:
        try:
            clicked_event_id = int(clicked_event)
            if clicked_event_id in event_ids:
                st.session_state[selector_key] = clicked_event_id
        except (TypeError, ValueError):
            pass

    st.caption(
        "这些自动事件点不是只看某一个音高或某一根频率线，而是综合看相邻频谱帧之间的结构变化。"
        "当前算法主要依据：频谱形态差异、谱流量、起音强度、能量突变，以及短、中、长三种时间尺度的共同证据。"
    )
    st.caption("事件点颜色现在表示“相似事件组”：如果两个事件的后态声音特征相似度达到约 90% 及以上，它们会用同一种颜色。")
    if str(result["config"].get("event_model_preset", "")) == "timbre_soundscape":
        st.caption("在“音色 / 声景模式”下，频谱形态差异这一项还会补充参考 spectral centroid、band energy ratio、高频占比和 flatness。")
    active_event_id = st.selectbox(
        "当前事件",
        options=event_ids,
        format_func=lambda event_id: event_label(event_ids, visible_annotations, event_id),
        key=selector_key,
    )

    active_row = visible_annotations.loc[visible_annotations["event_id"] == active_event_id].iloc[0]
    row_index = annotations.index[annotations["event_id"] == active_event_id][0]
    active_mode_label = EVENT_MODEL_PRESETS[preset_key]["label"]

    detail_col, right_col = st.columns([1.08, 0.92], gap="large")
    with detail_col:
        st.markdown("**事件详情**")
        current_time_label = str(active_row.get("manual_time_label", "")).strip() or str(active_row["time_label"])
        st.metric("当前时间", current_time_label)
        if current_time_label != str(active_row["time_label"]):
            st.caption(f"原始检测时间：{active_row['time_label']}")
        st.caption(
            f"交互层级 {active_row['interaction_priority_label']} | {active_row['similarity_group_label']} | "
            f"{active_row.get('evidence_label', '未分层')} {float(active_row.get('evidence_score', 0.0)):.2f} | "
            f"强度 {active_row['strength']:.3f} | 显著度 {active_row['prominence']:.3f} | 声道 {active_row['channel_bias']}"
        )
        if "evidence_summary" in active_row:
            st.write(f"检测证据：{active_row['evidence_summary']}")
        st.write(f"自动候选标签：`{active_row['auto_labels']}`")
        st.write(f"规则摘要：{active_row['descriptor']}")
        st.write(f"主导频率：{active_row['dominant_freqs']}")
        st.write(
            f"相似组信息：{active_row['similarity_group_label']}，组内共 {int(active_row['similarity_group_size'])} 个事件，"
            f"当前点与组内其他事件的最高相似度约为 {float(active_row['max_similarity_in_group']):.0%}"
        )

        default_manual = split_label_text(str(active_row["manual_labels"])) or split_label_text(str(active_row["auto_labels"]))
        default_custom = [label for label in default_manual if label not in result["event_label_vocab"]]
        default_manual_vocab = [label for label in default_manual if label in result["event_label_vocab"]]

        manual_vocab_key = f"manual_vocab_{analysis_key}_{active_event_id}"
        manual_custom_key = f"manual_custom_{analysis_key}_{active_event_id}"
        review_note_key = f"review_note_{analysis_key}_{active_event_id}"
        export_key = f"export_event_{analysis_key}_{active_event_id}"
        manual_time_key = f"manual_time_{analysis_key}_{active_event_id}"
        operation_result_key = f"operation_result_{analysis_key}_{active_event_id}"
        active_editor_key = f"active_event_editor_{analysis_key}"

        if st.session_state.get(active_editor_key) != active_event_id:
            st.session_state[manual_vocab_key] = default_manual_vocab
            st.session_state[manual_custom_key] = "、".join(default_custom)
            st.session_state[review_note_key] = str(active_row["review_notes"])
            st.session_state[export_key] = bool(active_row["export"])
            st.session_state[manual_time_key] = float(active_row.get("manual_time_sec", active_row["time_sec"]))
            st.session_state[operation_result_key] = str(active_row.get("operation_result", "")).strip() or "暂不处理"
            st.session_state[active_editor_key] = active_event_id

        chosen_vocab = st.multiselect(
            "人工标签",
            options=result["event_label_vocab"],
            key=manual_vocab_key,
        )
        custom_labels_text = st.text_input(
            "补充标签（可自定义，使用顿号或竖线分隔）",
            key=manual_custom_key,
        )
        review_notes = st.text_area(
            "人工备注",
            key=review_note_key,
            height=100,
        )
        export_flag = st.checkbox(
            "导出该事件",
            key=export_key,
        )
        st.markdown("**人工微调事件时间**")
        manual_time_columns = st.columns(3, gap="small")
        if manual_time_columns[0].button("前移 0.2 秒", key=f"shift_left_small_{analysis_key}_{active_event_id}", use_container_width=True):
            st.session_state[manual_time_key] = max(0.0, float(st.session_state.get(manual_time_key, active_row["time_sec"])) - 0.2)
        if manual_time_columns[1].button("回到原始时间", key=f"reset_time_{analysis_key}_{active_event_id}", use_container_width=True):
            st.session_state[manual_time_key] = float(active_row["time_sec"])
        if manual_time_columns[2].button("后移 0.2 秒", key=f"shift_right_small_{analysis_key}_{active_event_id}", use_container_width=True):
            st.session_state[manual_time_key] = min(float(result["duration_sec"]), float(st.session_state.get(manual_time_key, active_row["time_sec"])) + 0.2)

        manual_time_sec = st.number_input(
            "人工时间（秒）",
            min_value=0.0,
            max_value=float(result["duration_sec"]),
            step=0.05,
            key=manual_time_key,
            format="%.2f",
        )
        operation_result = st.selectbox(
            "你的操作结果",
            options=OPERATION_RESULT_OPTIONS,
            key=operation_result_key,
        )

        merged_labels = chosen_vocab + split_label_text(custom_labels_text)
        row_for_log = active_row.copy()
        row_for_log["manual_labels"] = join_labels(merged_labels)
        annotations.at[row_index, "manual_labels"] = join_labels(merged_labels)
        annotations.at[row_index, "review_notes"] = review_notes
        annotations.at[row_index, "export"] = export_flag
        annotations.at[row_index, "manual_time_sec"] = float(manual_time_sec)
        annotations.at[row_index, "manual_time_label"] = format_seconds(float(manual_time_sec))
        annotations.at[row_index, "operation_result"] = str(operation_result)
        annotations.at[row_index, "operation_log"] = format_operation_log(
            mode_label=active_mode_label,
            row=row_for_log,
            operation_result=str(operation_result),
            manual_time_sec=float(manual_time_sec),
            review_notes=str(review_notes),
        )

        st.markdown("**当前操作日志（可复制）**")
        st.text_area(
            "记录区",
            value=str(annotations.at[row_index, "operation_log"]),
            height=200,
            key=f"operation_log_view_{analysis_key}_{active_event_id}",
        )

    with right_col:
        st.markdown("**局部试听**")
        preview_radius = st.slider(
            "试听窗口（事件前后秒数）",
            min_value=1.0,
            max_value=15.0,
            value=4.0,
            step=0.5,
            key=f"preview_radius_{analysis_key}",
        )
        event_time = float(annotations.at[row_index, "manual_time_sec"])
        clip_start = max(0.0, event_time - preview_radius)
        clip_end = min(float(result["duration_sec"]), event_time + preview_radius)
        st.caption(f"试听区间：{format_seconds(clip_start)} - {format_seconds(clip_end)}")
        clip_signature = (
            analysis_key,
            round(clip_start, 4),
            round(clip_end, 4),
            int(result["sr"]),
        )
        clip_cache = st.session_state.get("_active_audio_event_clip")
        if not isinstance(clip_cache, dict) or clip_cache.get("signature") != clip_signature:
            clip_cache = {
                "signature": clip_signature,
                "wav": build_audio_excerpt_wav(
                    result["selected_audio"],
                    int(result["sr"]),
                    clip_start,
                    clip_end,
                    channel_mode="mix",
                ),
            }
            st.session_state["_active_audio_event_clip"] = clip_cache
        clip_bytes = clip_cache["wav"]

        st.markdown("**同步局部波形**")
        waveform_scale = st.slider(
            "波形放大倍数",
            min_value=1.0,
            max_value=8.0,
            value=3.0,
            step=0.5,
            key=f"waveform_scale_{analysis_key}",
            help="只影响波形显示，不影响音频播放和分析结果。",
        )
        components.html(
            build_synced_waveform_player_html(
                audio_bytes=clip_bytes,
                samples=result["selected_audio"],
                sr=int(result["sr"]),
                clip_start_sec=clip_start,
                clip_end_sec=clip_end,
                event_time_sec=event_time,
                amplitude_scale=float(waveform_scale),
            ),
            height=320,
            scrolling=False,
        )

        st.markdown("**局部频谱区域图**")
        st.caption(
            "局部频谱以当前试听窗口的峰值归一化为 0 dB，适合观察事件内部结构；"
            "跨段落比较绝对能量时，请结合整曲频谱和 RMS。深色高对比色阶中，"
            "紫蓝表示较弱能量，青白至黄绿表示较强能量。"
        )
        local_spec_max_freq = st.select_slider(
            "局部频谱最高显示频率",
            options=[2000, 5000, 8000, 10000, 16000, 22050],
            value=10000,
            key=f"local_spec_max_freq_{analysis_key}",
        )
        local_spectrum_signature = (
            analysis_key,
            round(event_time, 4),
            round(float(preview_radius), 3),
            int(local_spec_max_freq),
            int(result["config"]["n_fft"]),
            int(result["config"]["hop_length"]),
            float(result["config"]["spectrogram_dynamic_range_db"]),
            str(result.get("display_frequency_scale", "linear")),
        )
        local_spectrum_cache = st.session_state.get("_active_audio_local_spectrum")
        if (
            not isinstance(local_spectrum_cache, dict)
            or local_spectrum_cache.get("signature") != local_spectrum_signature
        ):
            local_spectrum_cache = {
                "signature": local_spectrum_signature,
                "png": figure_to_png_bytes(
                    plot_local_spectrogram(
                        result["spectrogram_db"],
                        result["spectrogram_times"],
                        result["spectrogram_freqs"],
                        center_time=event_time,
                        window_radius_sec=preview_radius,
                        highlight_time=event_time,
                        max_freq_hz=float(local_spec_max_freq),
                        samples=result["selected_audio"],
                        sr=int(result["sr"]),
                        n_fft=int(result["config"]["n_fft"]),
                        hop_length=int(result["config"]["hop_length"]),
                        dynamic_range_db=float(
                            result["config"]["spectrogram_dynamic_range_db"]
                        ),
                        frequency_scale=str(
                            result.get("display_frequency_scale", "linear")
                        ),
                    ),
                    dpi=150,
                ),
            }
            st.session_state["_active_audio_local_spectrum"] = local_spectrum_cache
        st.image(local_spectrum_cache["png"], width="stretch")

    st.markdown("**事件标注列表**")
    editor_view = visible_annotations[
        [
            "export",
            "event_id",
            "time_label",
            "interaction_priority_label",
            "evidence_label",
            "evidence_score",
            "multiscale_consensus",
            "similarity_group_label",
            "is_major_boundary",
            "auto_labels",
            "manual_labels",
            "review_notes",
            "descriptor",
            "channel_bias",
        ]
    ].copy()
    edited = st.data_editor(
        editor_view,
        width="stretch",
        hide_index=True,
        column_config={
            "export": st.column_config.CheckboxColumn("导出"),
            "event_id": st.column_config.NumberColumn("事件", disabled=True),
            "time_label": st.column_config.TextColumn("时间", disabled=True),
            "interaction_priority_label": st.column_config.TextColumn("交互层级", disabled=True),
            "evidence_label": st.column_config.TextColumn("证据层级", disabled=True),
            "evidence_score": st.column_config.NumberColumn("证据分数", disabled=True, format="%.2f"),
            "multiscale_consensus": st.column_config.NumberColumn("多尺度共识", disabled=True, format="%.0%%"),
            "similarity_group_label": st.column_config.TextColumn("相似组", disabled=True),
            "is_major_boundary": st.column_config.CheckboxColumn("候选边界", disabled=True),
            "auto_labels": st.column_config.TextColumn("自动标签", disabled=True, width="medium"),
            "manual_labels": st.column_config.TextColumn("人工标签", width="medium"),
            "review_notes": st.column_config.TextColumn("备注", width="large"),
            "descriptor": st.column_config.TextColumn("规则摘要", disabled=True, width="large"),
            "channel_bias": st.column_config.TextColumn("声道", disabled=True),
        },
        key=f"event_editor_{analysis_key}",
    )

    for edited_row in edited.itertuples(index=False):
        target_index = annotations.index[annotations["event_id"] == int(edited_row.event_id)][0]
        annotations.at[target_index, "export"] = bool(edited_row.export)
        annotations.at[target_index, "manual_labels"] = str(edited_row.manual_labels)
        annotations.at[target_index, "review_notes"] = str(edited_row.review_notes)

    log_rows = annotations.loc[
        annotations.apply(
            lambda row: (
                str(row.get("operation_result", "")).strip() not in {"", "暂不处理"}
                or bool(str(row.get("review_notes", "")).strip())
                or abs(float(row.get("manual_time_sec", row.get("time_sec", 0.0))) - float(row.get("time_sec", 0.0))) > 1e-6
            ),
            axis=1,
        )
    ].sort_values("time_sec")
    if not log_rows.empty:
        combined_log = "\n\n---\n\n".join(
            str(text).strip()
            for text in log_rows["operation_log"].tolist()
            if str(text).strip()
        )
        if combined_log:
            st.markdown("**全部操作记录（可复制）**")
            st.text_area(
                "汇总记录区",
                value=combined_log,
                height=260,
                key=f"combined_operation_log_{analysis_key}",
            )

    return annotations, active_event_id


def render_feature_chart(
    result: dict[str, object],
    analysis_key: str,
) -> None:
    feature_options = list(result["feature_column_labels"].keys())
    default_features = [
        feature_name
        for feature_name in [
            "rms",
            "spectral_centroid_hz",
            "band_energy_ratio",
            "high_band_ratio",
            "rolloff_hz",
            "flatness",
            "spectral_flux",
            "onset_strength",
            "novelty",
        ]
        if feature_name in feature_options
    ]

    selected_features = st.multiselect(
        "特征曲线",
        options=feature_options,
        default=default_features,
        format_func=lambda value: result["feature_column_labels"].get(value, value),
        key=f"feature_select_{analysis_key}",
    )
    normalize_features = st.toggle("归一化显示特征曲线", value=True, key=f"normalize_{analysis_key}")
    if len(result["feature_table"]) > 6_000:
        st.caption("长作品的特征图使用时间聚合以保证交互流畅；下载的特征曲线 CSV 仍保留完整帧数据。")

    st.altair_chart(
        build_feature_curve_chart(
            result["feature_table"],
            selected_features,
            result["feature_column_labels"],
            normalize=normalize_features,
        ),
        width="stretch",
    )
