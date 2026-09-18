from __future__ import annotations

import hashlib
import json

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from spectral_tool.analysis import AnalysisConfig, analyze_audio
from spectral_tool.models.presets import EVENT_MODEL_PRESETS, novelty_explanation
from spectral_tool.state.audio_state import (
    build_analysis_signature,
    filter_event_annotations,
    init_event_annotations,
    init_section_annotations,
    source_sha256,
)
from spectral_tool.ui.event_editor import render_event_editor, render_feature_chart
from spectral_tool.ui.section_editor import render_section_editor
from spectral_tool.visualization import (
    build_summary_text,
    build_synced_overview_player_html,
    figure_to_png_bytes,
    plot_event_density,
    plot_novelty,
    plot_spectrogram,
    plot_waveform,
)

_EMBEDDED_OVERVIEW_MAX_BYTES = 24 * 1024 * 1024


def _json_records(frame: object) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def render_audio_workspace(sidebar_values: dict[str, float | int | str | None]) -> None:
    st.markdown("### 音频 / 频谱事件分析工作台")

    uploaded_file = st.file_uploader(
        "上传音频文件（支持 WAV、FLAC、AIFF、MP3、M4A）",
        type=["wav", "flac", "aif", "aiff", "ogg", "mp3", "m4a"],
    )

    st.markdown(
        """
这个版本已经贴近本地 MVP 的完整工作流：

- 自动提取 `RMS / centroid / band energy ratio / high-frequency ratio / rolloff / flatness / flux / onset strength / novelty`
- 自动检测变化点、候选边界和新事件位置
- 点击自动标记点可显示精确时间
- 支持局部试听与人工改标签
- 支持导出编辑后的 `CSV / JSON`
"""
    )

    if uploaded_file is None:
        st.info("上传一段音频后，系统会自动生成频谱图、特征曲线、事件列表和可编辑标注。")
        st.stop()

    uploaded_size = int(getattr(uploaded_file, "size", 0) or 0)
    if uploaded_size <= 0:
        st.error("上传的音频文件为空，请重新选择有效文件。")
        st.stop()
    use_compact_overview = uploaded_size > _EMBEDDED_OVERVIEW_MAX_BYTES
    uploaded_file.seek(0)
    uploaded_mime = getattr(uploaded_file, "type", None) or "audio/wav"
    st.audio(uploaded_file, format=uploaded_mime)
    uploaded_file.seek(0)
    if use_compact_overview:
        st.info(
            "已启用长时程整曲模式：内部按连续音频块计算频谱以降低内存占用，"
            "事件、状态和段落再现仍在整首作品的统一时间轴上比较。"
        )

    config = AnalysisConfig(
        target_sr=sidebar_values["target_sr"],
        n_fft=sidebar_values["n_fft"],
        hop_length=sidebar_values["hop_length"],
        smooth_sigma=sidebar_values["smooth_sigma"],
        threshold_sigma=sidebar_values["threshold_sigma"],
        prominence_sigma=sidebar_values["prominence_sigma"],
        min_event_distance_sec=sidebar_values["min_event_distance"],
        context_window_sec=sidebar_values["context_window"],
        novelty_weight_cosine=sidebar_values["cosine_weight"],
        novelty_weight_flux=sidebar_values["flux_weight"],
        novelty_weight_onset=sidebar_values["onset_weight"],
        novelty_weight_rms=sidebar_values["rms_weight"],
        event_model_preset=str(sidebar_values["preset_key"]),
        multiscale_enabled=bool(sidebar_values["multiscale_enabled"]),
        multiscale_weight=float(sidebar_values["multiscale_weight"]),
        spectrogram_dynamic_range_db=float(sidebar_values["spectrogram_dynamic_range_db"]),
    )
    frequency_scale = (
        "log" if str(sidebar_values.get("frequency_scale", "linear")) == "log" else "linear"
    )

    channel_mode = str(sidebar_values["channel_mode"])
    preset_key = str(sidebar_values["preset_key"])
    upload_file_id = getattr(uploaded_file, "file_id", None)
    upload_identity = (
        str(upload_file_id) if upload_file_id is not None else "",
        str(getattr(uploaded_file, "name", "")),
        uploaded_size,
        uploaded_mime,
    )
    try:
        digest_cache = st.session_state.get("_active_audio_source_digest")
        if (
            upload_file_id is not None
            and isinstance(digest_cache, dict)
            and digest_cache.get("identity") == upload_identity
            and digest_cache.get("sha256")
        ):
            audio_digest = str(digest_cache["sha256"])
        else:
            audio_digest = source_sha256(uploaded_file)
            if upload_file_id is not None:
                st.session_state["_active_audio_source_digest"] = {
                    "identity": upload_identity,
                    "sha256": audio_digest,
                }
        analysis_key = build_analysis_signature(
            uploaded_file,
            config,
            channel_mode,
            audio_sha256=audio_digest,
        )
    except (OSError, TypeError, ValueError) as error:
        st.error("无法读取上传文件，请确认文件没有损坏，并重新上传。")
        st.caption(str(error))
        st.stop()

    cached_result = st.session_state.get("_active_audio_analysis_result")
    if (
        st.session_state.get("_active_audio_analysis_key") == analysis_key
        and isinstance(cached_result, dict)
    ):
        result = cached_result
    else:
        try:
            with st.spinner("正在分析频谱变化、提取特征并构建候选标注..."):
                result = analyze_audio(
                    uploaded_file,
                    config=config,
                    channel_mode=channel_mode,
                    source_sha256=audio_digest,
                )
        except MemoryError:
            st.error("可用内存不足，建议把目标采样率设为 22050 或 32000 Hz 后重试。")
            st.stop()
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            st.error("音频分析未完成。请检查文件格式，或降低采样率与频谱窗长后重试。")
            st.caption(str(error))
            st.stop()
        st.session_state["_active_audio_analysis_key"] = analysis_key
        st.session_state["_active_audio_analysis_result"] = result
        st.session_state.pop("_active_audio_export_bundle", None)
        st.session_state.pop("_active_audio_event_clip", None)
        st.session_state.pop("_active_audio_local_spectrum", None)
        st.session_state.pop("_active_audio_overview_figures", None)
        st.session_state.pop("_active_audio_overview_html", None)
        st.session_state.pop("_active_audio_novelty_figure", None)
    result["display_frequency_scale"] = frequency_scale

    event_state_key = init_event_annotations(result, analysis_key)
    section_state_key = init_section_annotations(result, analysis_key)
    event_annotations = st.session_state[event_state_key].copy()
    section_annotations = st.session_state[section_state_key].copy()
    active_filter_labels = st.session_state.get(f"event_label_filter_{analysis_key}", [])
    overview_annotations = filter_event_annotations(event_annotations, active_filter_labels)
    if overview_annotations.empty:
        overview_event_times = result["peak_times"][:0]
        overview_event_labels = np.array([], dtype=int)
        overview_major_events = np.array([], dtype=bool)
    else:
        overview_time_column = (
            "manual_time_sec"
            if "manual_time_sec" in overview_annotations.columns
            else "time_sec"
        )
        overview_event_times = overview_annotations[overview_time_column].to_numpy(dtype=float)
        overview_event_labels = overview_annotations["event_id"].to_numpy(dtype=int)
        overview_major_events = overview_annotations["is_major_boundary"].to_numpy(dtype=bool)
    overview_figure_signature = (
        analysis_key,
        frequency_scale,
        tuple(
            (int(event_id), round(float(event_time), 4), bool(is_major))
            for event_id, event_time, is_major in zip(
                overview_event_labels,
                overview_event_times,
                overview_major_events,
                strict=True,
            )
        ),
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("时长", f"{result['duration_sec']:.2f} 秒")
    metric_2.metric("采样率", f"{result['sr']} Hz")
    metric_3.metric("检测事件数", int(len(result["event_table"])))
    metric_4.metric(
        "候选边界数",
        int(result["event_table"]["is_major_boundary"].sum()) if not result["event_table"].empty else 0,
    )

    st.subheader("机器摘要")
    st.text(build_summary_text(result["summary_lines"]))

    tab_1, tab_2, tab_3, tab_4 = st.tabs(["总览", "交互分析", "标注编辑", "导出"])

    with tab_1:
        st.markdown("**同步总览播放器**")
        max_analyzable_hz = float(result["sr"]) / 2.0
        st.caption(
            "这一块是不带自动事件编号线的纯频谱总览。播放时，波形和频谱会一起跟着播放头移动，并自动聚焦到播放头附近的局部时间窗口。"
            f"当前分析采样率为 {int(result['sr'])} Hz，所以机器最多能客观分析到约 {int(max_analyzable_hz)} Hz。"
            "如果原文件本身只有 44.1kHz，那么 25kHz 以上的信息并不在文件里。"
        )
        if use_compact_overview:
            st.caption(
                "大文件使用页面顶部的原始音频播放器；此处不再把整首音频复制为 Base64，"
                "从而为完整频谱与全曲相似度分析保留内存。"
            )
        else:
            overview_html_cache = st.session_state.get("_active_audio_overview_html")
            overview_html_signature = (analysis_key, frequency_scale)
            if (
                not isinstance(overview_html_cache, dict)
                or overview_html_cache.get("signature") != overview_html_signature
            ):
                uploaded_file.seek(0)
                audio_bytes = uploaded_file.read()
                uploaded_file.seek(0)
                overview_html_cache = {
                    "signature": overview_html_signature,
                    "html": build_synced_overview_player_html(
                        audio_bytes=audio_bytes,
                        audio_mime=uploaded_mime,
                        samples=result["selected_audio"],
                        sr=int(result["sr"]),
                        spectrogram_db=result["spectrogram_db"],
                        spectrogram_times=result["spectrogram_times"],
                        spectrogram_freqs=result["spectrogram_freqs"],
                        duration_sec=float(result["duration_sec"]),
                        initial_wave_amplitude_max=None,
                        initial_freq_max_hz=float(result["spectrogram_freqs"][-1])
                        if len(result["spectrogram_freqs"])
                        else max_analyzable_hz,
                        frequency_scale=frequency_scale,
                    ),
                }
                st.session_state["_active_audio_overview_html"] = overview_html_cache
            components.html(
                overview_html_cache["html"],
                height=980,
                scrolling=False,
            )

        st.markdown("**增强频谱总览**")
        st.caption(
            "采用深色高对比频谱：窄带峰值优先保留，时间方向使用 RMS 聚合；"
            f"当前为{'对数' if frequency_scale == 'log' else '线性'}频率刻度。"
        )
        if active_filter_labels:
            st.caption(
                "总览当前只高亮这些标签对应的事件："
                + "、".join(active_filter_labels)
                + f"（{len(overview_event_times)} 个事件）"
            )
        overview_figure_cache = st.session_state.get("_active_audio_overview_figures")
        if (
            not isinstance(overview_figure_cache, dict)
            or overview_figure_cache.get("signature") != overview_figure_signature
        ):
            overview_figure_cache = {
                "signature": overview_figure_signature,
                "spectrogram": figure_to_png_bytes(
                    plot_spectrogram(
                        result["spectrogram_db"],
                        result["spectrogram_times"],
                        result["spectrogram_freqs"],
                        overview_event_times,
                        event_labels=overview_event_labels,
                        major_event_mask=overview_major_events,
                        dynamic_range_db=float(
                            result["config"]["spectrogram_dynamic_range_db"]
                        ),
                        frequency_scale=frequency_scale,
                    ),
                    dpi=150,
                ),
                "waveform": figure_to_png_bytes(
                    plot_waveform(
                        result["selected_audio"],
                        result["sr"],
                        overview_event_times,
                        major_event_mask=overview_major_events,
                    ),
                    dpi=150,
                ),
                "density": figure_to_png_bytes(
                    plot_event_density(
                        overview_event_times,
                        result["duration_sec"],
                        major_event_mask=overview_major_events,
                    ),
                    dpi=150,
                ),
            }
            st.session_state["_active_audio_overview_figures"] = overview_figure_cache
        st.image(overview_figure_cache["spectrogram"], width="stretch")
        st.image(overview_figure_cache["waveform"], width="stretch")
        st.image(overview_figure_cache["density"], width="stretch")

        st.markdown("**声景状态指纹**")
        st.caption(
            "每个段落被表示为一组可比较的状态维度。带 `_z` 的列表示该段相对于本作品整体中位状态的偏离，"
            "适合用于同一文件内部的结构比较；原始值可保留为分析依据。"
        )
        fingerprint_table = result["section_fingerprint_table"]
        if fingerprint_table.empty:
            st.info("当前结果没有形成可比较的段落状态指纹。")
        else:
            st.dataframe(fingerprint_table, width="stretch", hide_index=True)

        recurrence_table = result["state_similarity_table"]
        recurrence_candidates = (
            recurrence_table.loc[recurrence_table["is_recurrence_candidate"]].copy()
            if not recurrence_table.empty
            else recurrence_table
        )
        with st.expander(f"声景状态再现候选（{len(recurrence_candidates)} 对）", expanded=False):
            if recurrence_candidates.empty:
                st.caption("当前段落之间没有达到阈值的状态再现候选。")
            else:
                st.dataframe(recurrence_candidates, width="stretch", hide_index=True)

    with tab_2:
        st.markdown("**自动事件点的判定依据**")
        st.caption(f"当前模式：{EVENT_MODEL_PRESETS[preset_key]['label']}")
        st.markdown(novelty_explanation(config))
        novelty_figure_cache = st.session_state.get("_active_audio_novelty_figure")
        if (
            not isinstance(novelty_figure_cache, dict)
            or novelty_figure_cache.get("analysis_key") != analysis_key
        ):
            novelty_figure_cache = {
                "analysis_key": analysis_key,
                "png": figure_to_png_bytes(
                    plot_novelty(
                        result["times"],
                        result["novelty"],
                        result["threshold"],
                        result["peak_indices"],
                        major_event_mask=result["event_table"]["is_major_boundary"].to_numpy(
                            dtype=bool
                        )
                        if not result["event_table"].empty
                        else np.array([], dtype=bool),
                    ),
                    dpi=150,
                ),
            }
            st.session_state["_active_audio_novelty_figure"] = novelty_figure_cache
        st.image(novelty_figure_cache["png"], width="stretch")

        render_feature_chart(result, analysis_key)
        event_annotations, _ = render_event_editor(event_annotations, result, analysis_key, preset_key)
        st.session_state[event_state_key] = event_annotations

    with tab_3:
        if event_annotations.empty:
            st.warning("当前参数下没有检测到可编辑的自动事件。")
        else:
            st.markdown("**当前事件修订结果**")
            st.dataframe(
                event_annotations[
                    [
                        "export",
                        "event_id",
                        "time_label",
                        "auto_labels",
                        "manual_labels",
                        "review_notes",
                        "is_major_boundary",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

        section_annotations = render_section_editor(section_annotations, analysis_key)
        st.session_state[section_state_key] = section_annotations

    with tab_4:
        exported_events = event_annotations.copy()
        exported_sections = section_annotations.copy()
        if not exported_events.empty:
            exported_events = exported_events.loc[exported_events["export"]].copy()
        if not exported_sections.empty:
            exported_sections = exported_sections.loc[exported_sections["export"]].copy()

        export_time_column = "manual_time_sec" if "manual_time_sec" in exported_events.columns else "time_sec"
        exported_event_times = (
            exported_events[export_time_column].to_numpy(dtype=float)
            if not exported_events.empty
            else result["peak_times"][:0]
        )
        exported_event_ids = (
            exported_events["event_id"].to_numpy(dtype=int)
            if not exported_events.empty
            else np.array([], dtype=int)
        )
        exported_major_events = (
            exported_events["is_major_boundary"].to_numpy(dtype=bool)
            if not exported_events.empty
            else np.array([], dtype=bool)
        )
        event_figure_signature = tuple(
            (int(event_id), round(float(event_time), 6), bool(is_major))
            for event_id, event_time, is_major in zip(
                exported_event_ids,
                exported_event_times,
                exported_major_events,
                strict=True,
            )
        )
        annotation_digest = hashlib.sha256(
            (
                analysis_key
                + exported_events.to_json(orient="split", date_format="iso")
                + exported_sections.to_json(orient="split", date_format="iso")
                + repr(event_figure_signature)
                + frequency_scale
            ).encode("utf-8")
        ).hexdigest()
        export_cache = st.session_state.get("_active_audio_export_bundle")
        export_is_current = (
            isinstance(export_cache, dict)
            and export_cache.get("signature") == annotation_digest
        )

        st.caption(
            "为避免每次编辑事件时重复生成大型文件，CSV、JSON 与高分辨率 PNG 只在点击后统一准备。"
        )
        prepare_label = "重新生成导出文件" if export_cache else "准备全部导出文件"
        if st.button(
            prepare_label,
            key=f"prepare_audio_exports_{analysis_key}",
            type="primary",
        ):
            with st.spinner("正在整理表格、JSON 与高分辨率频谱图..."):
                dynamic_range_db = float(
                    result["config"]["spectrogram_dynamic_range_db"]
                )
                export_cache = {
                    "signature": annotation_digest,
                    "event_csv": exported_events.to_csv(index=False).encode("utf-8-sig"),
                    "section_csv": exported_sections.to_csv(index=False).encode("utf-8-sig"),
                    "fingerprint_csv": result["section_fingerprint_table"]
                    .to_csv(index=False)
                    .encode("utf-8-sig"),
                    "state_similarity_csv": result["state_similarity_table"]
                    .to_csv(index=False)
                    .encode("utf-8-sig"),
                    "feature_csv": result["feature_table"]
                    .to_csv(index=False)
                    .encode("utf-8-sig"),
                    "summary_text": build_summary_text(result["summary_lines"]).encode(
                        "utf-8"
                    ),
                    "pure_spectrogram_png": figure_to_png_bytes(
                        plot_spectrogram(
                            result["spectrogram_db"],
                            result["spectrogram_times"],
                            result["spectrogram_freqs"],
                            result["peak_times"][:0],
                            dynamic_range_db=dynamic_range_db,
                            frequency_scale=frequency_scale,
                        )
                    ),
                    "event_spectrogram_png": figure_to_png_bytes(
                        plot_spectrogram(
                            result["spectrogram_db"],
                            result["spectrogram_times"],
                            result["spectrogram_freqs"],
                            exported_event_times,
                            event_labels=exported_event_ids,
                            major_event_mask=exported_major_events,
                            dynamic_range_db=dynamic_range_db,
                            frequency_scale=frequency_scale,
                        )
                    ),
                }
                export_cache["analysis_json"] = json.dumps(
                    {
                        "config": result["config"],
                        "analysis_metadata": result["analysis_metadata"],
                        "duration_sec": result["duration_sec"],
                        "sr": result["sr"],
                        "channel_mode": result["channel_mode"],
                        "summary_lines": result["summary_lines"],
                        "events": _json_records(exported_events),
                        "sections": _json_records(exported_sections),
                        "section_fingerprints": _json_records(
                            result["section_fingerprint_table"]
                        ),
                        "state_similarities": _json_records(
                            result["state_similarity_table"]
                        ),
                        "feature_table": _json_records(result["feature_table"]),
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                st.session_state["_active_audio_export_bundle"] = export_cache
                export_is_current = True

        if export_cache and not export_is_current:
            st.warning("事件或段落标注已经变化，请重新生成后再下载，避免导出旧结果。")
        elif export_is_current:
            st.download_button(
                "下载编辑后事件表 CSV",
                data=export_cache["event_csv"],
                file_name="annotated_events.csv",
                mime="text/csv",
            )
            st.download_button(
                "下载编辑后段落表 CSV",
                data=export_cache["section_csv"],
                file_name="annotated_sections.csv",
                mime="text/csv",
            )
            st.download_button(
                "下载声景状态指纹 CSV",
                data=export_cache["fingerprint_csv"],
                file_name="soundscape_state_fingerprints.csv",
                mime="text/csv",
            )
            st.download_button(
                "下载状态相似度 CSV",
                data=export_cache["state_similarity_csv"],
                file_name="soundscape_state_similarities.csv",
                mime="text/csv",
            )
            st.download_button(
                "下载特征曲线 CSV",
                data=export_cache["feature_csv"],
                file_name="feature_table.csv",
                mime="text/csv",
            )
            st.download_button(
                "下载纯频谱图 PNG（高分辨率）",
                data=export_cache["pure_spectrogram_png"],
                file_name="spectrogram_clean.png",
                mime="image/png",
            )
            st.download_button(
                "下载事件标注频谱图 PNG（高分辨率）",
                data=export_cache["event_spectrogram_png"],
                file_name="spectrogram_events.png",
                mime="image/png",
            )
            st.download_button(
                "下载摘要 TXT",
                data=export_cache["summary_text"],
                file_name="summary.txt",
                mime="text/plain",
            )
            st.download_button(
                "下载完整分析 JSON",
                data=export_cache["analysis_json"],
                file_name="analysis.json",
                mime="application/json",
            )

    st.markdown(
        """
提示：

- 事件太少：降低“检测阈值强度”或缩短“最小事件间隔”
- 事件太密：提高阈值或增大最小事件间隔
- 点击自动标记点后，右侧会直接显示精确时间与局部试听
- 人工修订后的标签会参与最终导出
"""
    )
