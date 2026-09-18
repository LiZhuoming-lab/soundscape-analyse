from __future__ import annotations

import streamlit as st

from spectral_tool.models.presets import EVENT_MODEL_PRESETS, format_weight_percent


def render_audio_sidebar() -> dict[str, float | int | str | None]:
    st.subheader("音频分析参数")
    st.caption("调整参数后点击底部按钮，避免拖动控件时反复分析整首音频。")
    with st.form("audio_analysis_parameters", border=False):
        preset_key = st.selectbox(
            "自动事件点识别模式",
            options=list(EVENT_MODEL_PRESETS.keys()),
            format_func=lambda value: EVENT_MODEL_PRESETS[value]["label"],
        )
        preset = EVENT_MODEL_PRESETS[preset_key]
        cosine_weight, flux_weight, onset_weight, rms_weight = preset["weights"]
        st.caption(preset["note"])
        st.caption(
            "当前预设权重："
            f"频谱形态差异 {format_weight_percent(cosine_weight)}% / "
            f"spectral flux {format_weight_percent(flux_weight)}% / "
            f"onset strength {format_weight_percent(onset_weight)}% / "
            f"RMS delta {format_weight_percent(rms_weight)}%"
        )
        if preset_key == "timbre_soundscape":
            st.caption(
                "当前模式会额外参考：spectral centroid / band energy ratio / "
                "高频占比 / flatness。"
            )
        channel_mode = st.selectbox(
            "分析通道",
            options=["mix", "left", "right"],
            format_func=lambda value: {
                "mix": "混合",
                "left": "左声道",
                "right": "右声道",
            }[value],
        )
        target_sr_option = st.selectbox(
            "目标采样率",
            options=["原始采样率", "44100", "32000", "22050"],
            index=2,
            help=(
                "云端整曲分析推荐 32000 Hz，可保留至 16 kHz 的频谱信息并显著降低内存；"
                "本地高频研究可选原始采样率。"
            ),
        )
        target_sr = None if target_sr_option == "原始采样率" else int(target_sr_option)
        n_fft = st.select_slider(
            "频谱窗长 n_fft",
            options=[1024, 2048, 4096, 8192],
            value=4096,
        )
        hop_length = st.select_slider(
            "帧移 hop_length",
            options=[256, 512, 1024, 2048],
            value=1024,
        )
        if preset.get("min_event_distance_sec") is None or preset_key == "balanced_default":
            min_event_distance = st.slider(
                "最小事件间隔（秒）",
                min_value=0.5,
                max_value=20.0,
                value=5.0,
                step=0.5,
            )
        else:
            min_event_distance = float(preset["min_event_distance_sec"])
            st.caption(f"当前预设自动使用：最小事件间隔 {min_event_distance:.1f} 秒")
        context_window = st.slider(
            "事件前后比较窗口（秒）",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
        )
        smooth_sigma = st.slider(
            "新颖度平滑强度",
            min_value=0.5,
            max_value=4.0,
            value=1.6,
            step=0.1,
        )
        if preset.get("threshold_sigma") is None or preset_key == "balanced_default":
            threshold_sigma = st.slider(
                "检测阈值强度",
                min_value=0.2,
                max_value=2.5,
                value=1.0,
                step=0.1,
            )
        else:
            threshold_sigma = float(preset["threshold_sigma"])
            st.caption(f"当前预设自动使用：检测阈值强度 {threshold_sigma:.1f}")
        prominence_sigma = st.slider(
            "峰值显著度强度",
            min_value=0.2,
            max_value=2.0,
            value=0.8,
            step=0.1,
        )
        with st.expander("方法学高级参数", expanded=False):
            multiscale_enabled = st.toggle(
                "启用多尺度共识检测",
                value=True,
                help="同时比较短、中、长三种频谱时间窗；用于降低单一窗长带来的误判。",
            )
            multiscale_weight = st.slider(
                "多尺度证据权重",
                min_value=0.0,
                max_value=0.7,
                value=0.35,
                step=0.05,
                disabled=not multiscale_enabled,
            )
            spectrogram_dynamic_range_db = st.slider(
                "频谱图动态范围（dB）",
                min_value=50,
                max_value=120,
                value=90,
                step=5,
                help="固定动态范围使不同作品的频谱图更便于横向比较。",
            )
            frequency_scale = st.selectbox(
                "频谱频率刻度",
                options=["linear", "log"],
                format_func=lambda value: {
                    "linear": "线性：观察全频段分布",
                    "log": "对数：展开低频与谐波细节",
                }[value],
                help="只改变频谱呈现，不改变事件检测和导出特征数值。",
            )
        st.form_submit_button(
            "应用参数并重新分析",
            type="primary",
            use_container_width=True,
        )

    return {
        "preset_key": preset_key,
        "channel_mode": channel_mode,
        "target_sr": target_sr,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "min_event_distance": min_event_distance,
        "context_window": context_window,
        "smooth_sigma": smooth_sigma,
        "threshold_sigma": threshold_sigma,
        "prominence_sigma": prominence_sigma,
        "cosine_weight": cosine_weight,
        "flux_weight": flux_weight,
        "onset_weight": onset_weight,
        "rms_weight": rms_weight,
        "multiscale_enabled": multiscale_enabled,
        "multiscale_weight": multiscale_weight,
        "spectrogram_dynamic_range_db": spectrogram_dynamic_range_db,
        "frequency_scale": frequency_scale,
    }
