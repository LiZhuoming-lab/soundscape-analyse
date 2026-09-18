from __future__ import annotations

import base64
import colorsys
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd

_CACHE_ROOT = Path(tempfile.gettempdir()) / "spectral_tool_cache"
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "mpl").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "mpl"))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "STHeiti",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "SimHei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from .analysis import _iter_magnitude_blocks, _magnitude_to_relative_db, format_seconds

_SPECTROGRAM_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "ianalyse_clarity",
    [
        (0.00, "#020308"),
        (0.10, "#090a20"),
        (0.25, "#211446"),
        (0.42, "#35266f"),
        (0.58, "#1652a3"),
        (0.74, "#008fcb"),
        (0.87, "#21d4e8"),
        (0.96, "#d7fff1"),
        (1.00, "#f4ff72"),
    ],
    N=256,
)


def _spectrogram_norm(dynamic_range_db: float) -> matplotlib.colors.Normalize:
    return matplotlib.colors.PowerNorm(
        gamma=0.72,
        vmin=-max(20.0, float(dynamic_range_db)),
        vmax=0.0,
        clip=True,
    )


def _prepare_frequency_display(
    spectrogram_db: np.ndarray,
    freqs: np.ndarray,
    frequency_scale: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(spectrogram_db)
    frequencies = np.asarray(freqs, dtype=np.float32)
    if frequency_scale != "log" or frequencies.size == 0:
        return values, frequencies
    positive_mask = frequencies >= 20.0
    if not np.any(positive_mask):
        positive_mask = frequencies > 0.0
    if not np.any(positive_mask):
        return values, frequencies
    return values[positive_mask, :], frequencies[positive_mask]


def _configure_frequency_axis(
    axis: plt.Axes,
    frequencies_hz: np.ndarray,
    frequency_scale: str,
) -> None:
    if frequency_scale != "log" or frequencies_hz.size == 0:
        axis.set_ylabel("频率（kHz）", labelpad=12)
        return

    minimum_hz = max(20.0, float(frequencies_hz[0]))
    maximum_hz = float(frequencies_hz[-1])
    tick_candidates = np.array(
        [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 16000, 22050],
        dtype=np.float64,
    )
    ticks_hz = tick_candidates[
        (tick_candidates >= minimum_hz) & (tick_candidates <= maximum_hz)
    ]
    axis.set_yscale("log")
    axis.set_ylim(minimum_hz / 1000.0, maximum_hz / 1000.0)
    axis.set_yticks(ticks_hz / 1000.0)
    axis.set_yticklabels(
        [f"{int(value)} Hz" if value < 1000 else f"{value / 1000:g} kHz" for value in ticks_hz]
    )
    axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axis.set_ylabel("频率（对数刻度）", labelpad=12)


def _downsample_waveform_extrema(
    samples: np.ndarray,
    sr: int,
    max_bins: int = 60_000,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.array([], dtype=np.float64), values
    if values.size <= max_bins * 2:
        times = np.arange(values.size, dtype=np.float64) / max(int(sr), 1)
        return times, values

    bin_size = max(1, int(np.ceil(values.size / max(max_bins, 1))))
    full_bin_count = values.size // bin_size
    if full_bin_count:
        blocks = values[: full_bin_count * bin_size].reshape(full_bin_count, bin_size)
        local_indices = np.column_stack(
            [np.argmin(blocks, axis=1), np.argmax(blocks, axis=1)]
        ).astype(np.int64)
        local_indices.sort(axis=1)
        starts = (np.arange(full_bin_count, dtype=np.int64) * bin_size)[:, None]
        point_indices = (starts + local_indices).reshape(-1)
    else:
        point_indices = np.array([], dtype=np.int64)

    tail_start = full_bin_count * bin_size
    if tail_start < values.size:
        tail = values[tail_start:]
        tail_indices = np.sort(
            np.array([int(np.argmin(tail)), int(np.argmax(tail))], dtype=np.int64)
        )
        point_indices = np.concatenate([point_indices, tail_start + tail_indices])

    times = point_indices.astype(np.float64) / max(int(sr), 1)
    return times, values[point_indices]


def _format_time_axis(axis: plt.Axes, duration_sec: float) -> None:
    if duration_sec < 120.0:
        axis.set_xlabel("时间（秒）")
        return

    axis.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda value, _position: f"{int(max(value, 0.0) // 60):02d}:{int(max(value, 0.0) % 60):02d}"
        )
    )
    axis.set_xlabel("时间（分:秒）")


def _draw_event_markers(
    axis: plt.Axes,
    event_times: np.ndarray,
    label_y: float | None = None,
    max_labels: int = 24,
    event_labels: np.ndarray | list[object] | None = None,
    major_event_mask: np.ndarray | list[bool] | None = None,
) -> None:
    times = np.asarray(event_times, dtype=float)
    if times.size == 0:
        return
    labels = (
        np.asarray(event_labels, dtype=object)
        if event_labels is not None
        else np.arange(1, times.size + 1)
    )
    if labels.size != times.size:
        labels = np.arange(1, times.size + 1)
    major_flags = (
        np.asarray(major_event_mask, dtype=bool)
        if major_event_mask is not None
        else np.zeros(times.size, dtype=bool)
    )
    if major_flags.size != times.size:
        major_flags = np.zeros(times.size, dtype=bool)

    line_alpha = 0.58 if times.size <= 40 else 0.30
    line_width = 0.9 if times.size <= 40 else 0.6
    for event_index, event_time in enumerate(times):
        is_major = bool(major_flags[event_index])
        axis.axvline(
            float(event_time),
            color=("#ffe66d" if label_y is not None else "#c44536")
            if is_major
            else ("white" if label_y is not None else "#d99058"),
            alpha=min(0.95, line_alpha + 0.22) if is_major else line_alpha,
            linewidth=line_width + 0.9 if is_major else line_width,
        )

    if label_y is None:
        return

    label_count = min(max_labels, times.size)
    label_indices = np.unique(np.linspace(0, times.size - 1, label_count, dtype=int))
    for event_index in label_indices:
        axis.text(
            float(times[event_index]),
            label_y,
            str(labels[event_index]),
            color="#151820" if major_flags[event_index] else "white",
            fontsize=7.5,
            ha="center",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.13",
                "facecolor": "#ffe66d" if major_flags[event_index] else "black",
                "alpha": 0.88 if major_flags[event_index] else 0.45,
                "edgecolor": "none",
            },
        )


def plot_waveform(
    samples: np.ndarray,
    sr: int,
    event_times: np.ndarray,
    major_event_mask: np.ndarray | list[bool] | None = None,
) -> plt.Figure:
    duration = samples.shape[-1] / sr
    time_axis, plotted_samples = _downsample_waveform_extrema(samples, sr)

    figure, axis = plt.subplots(figsize=(14, 3.5))
    axis.plot(time_axis, plotted_samples, linewidth=0.7, color="#1f5c99")
    _draw_event_markers(axis, event_times, major_event_mask=major_event_mask)

    axis.set_title("波形与事件标记")
    _format_time_axis(axis, duration)
    axis.set_ylabel("振幅")
    axis.set_xlim(0, duration)
    axis.grid(alpha=0.2, linestyle="--")
    figure.tight_layout()
    return figure


def plot_novelty(
    times: np.ndarray,
    novelty: np.ndarray,
    threshold: np.ndarray,
    peak_indices: np.ndarray,
    major_event_mask: np.ndarray | list[bool] | None = None,
) -> plt.Figure:
    duration = float(times[-1]) if times.size else 0.0
    figure, axis = plt.subplots(figsize=(14, 3.5))
    axis.plot(times, novelty, color="#2a9d8f", linewidth=1.2, label="新颖度曲线")
    axis.plot(times, threshold, color="#e76f51", linewidth=1.0, linestyle="--", label="检测阈值")
    if peak_indices.size:
        major_flags = (
            np.asarray(major_event_mask, dtype=bool)
            if major_event_mask is not None
            else np.zeros(peak_indices.size, dtype=bool)
        )
        if major_flags.size != peak_indices.size:
            major_flags = np.zeros(peak_indices.size, dtype=bool)
        ordinary_indices = peak_indices[~major_flags]
        major_indices = peak_indices[major_flags]
        point_size = 28 if peak_indices.size > 40 else 35
        point_alpha = 0.78 if peak_indices.size > 40 else 0.95
        if ordinary_indices.size:
            axis.scatter(
                times[ordinary_indices],
                novelty[ordinary_indices],
                color="#9c3d54",
                s=point_size,
                alpha=point_alpha,
                label="普通事件",
                zorder=3,
            )
        if major_indices.size:
            axis.scatter(
                times[major_indices],
                novelty[major_indices],
                color="#c44536",
                edgecolors="white",
                linewidths=0.7,
                marker="D",
                s=point_size + 28,
                alpha=0.98,
                label="候选边界",
                zorder=4,
            )

    axis.set_title("频谱新颖度检测")
    _format_time_axis(axis, duration)
    axis.set_ylabel("新颖度")
    if duration > 0:
        axis.set_xlim(0, duration)
    axis.grid(alpha=0.2, linestyle="--")
    axis.legend(loc="upper right")
    figure.tight_layout()
    return figure


def plot_spectrogram(
    spectrogram_db: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    event_times: np.ndarray,
    event_labels: np.ndarray | list[object] | None = None,
    major_event_mask: np.ndarray | list[bool] | None = None,
    dynamic_range_db: float = 90.0,
    frequency_scale: str = "linear",
) -> plt.Figure:
    dynamic_range = max(20.0, float(dynamic_range_db))
    display_db, display_freqs = _prepare_frequency_display(
        spectrogram_db,
        freqs,
        frequency_scale,
    )
    figure, axis = plt.subplots(figsize=(14.8, 6.2))
    mesh = axis.pcolormesh(
        times,
        display_freqs / 1000.0,
        display_db,
        shading="auto",
        cmap=_SPECTROGRAM_CMAP,
        norm=_spectrogram_norm(dynamic_range),
        rasterized=True,
    )

    upper_y = (display_freqs[-1] / 1000.0) * 0.98 if display_freqs.size else 1.0
    _draw_event_markers(
        axis,
        event_times,
        label_y=upper_y,
        event_labels=event_labels,
        major_event_mask=major_event_mask,
    )

    axis.set_title("频谱图与事件编号" if len(event_times) else "频谱图")
    if times.size:
        time_step = float(np.median(np.diff(times))) if times.size > 1 else 0.0
        duration = max(0.0, float(times[-1]) + time_step / 2.0)
        axis.set_xlim(0, duration)
    else:
        duration = 0.0
    _format_time_axis(axis, duration)
    _configure_frequency_axis(axis, display_freqs, frequency_scale)
    axis.grid(axis="y", color="white", alpha=0.10, linewidth=0.55)
    colorbar = figure.colorbar(mesh, ax=axis, format="%+2.0f dB", pad=0.02, fraction=0.035)
    colorbar.set_label("相对幅度（dB）")
    figure.subplots_adjust(left=0.13, right=0.92, bottom=0.11, top=0.92)
    return figure


def plot_local_waveform(
    samples: np.ndarray,
    sr: int,
    center_time: float,
    window_radius_sec: float,
    highlight_time: float | None = None,
) -> plt.Figure:
    if samples.size == 0:
        figure, axis = plt.subplots(figsize=(8.2, 2.6))
        axis.text(0.5, 0.5, "没有可显示的局部波形", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        figure.tight_layout()
        return figure

    start_sec = max(0.0, float(center_time - window_radius_sec))
    end_sec = min(float(samples.shape[-1] / sr), float(center_time + window_radius_sec))
    start_sample = max(0, int(start_sec * sr))
    end_sample = min(samples.shape[-1], max(start_sample + 1, int(end_sec * sr)))

    local_samples = samples[start_sample:end_sample]
    local_time = np.linspace(start_sec, end_sec, local_samples.shape[0], endpoint=False)

    figure, axis = plt.subplots(figsize=(8.2, 2.6))
    axis.plot(local_time, local_samples, linewidth=0.8, color="#1f5c99")
    axis.axhline(0.0, color="#7a7a7a", linewidth=0.6, alpha=0.6)
    if highlight_time is None:
        highlight_time = center_time
    axis.axvline(float(highlight_time), color="#d1495b", linewidth=1.2, alpha=0.95)
    axis.set_title(f"事件附近局部波形：{center_time:.2f}s")
    axis.set_xlabel("时间（秒）")
    axis.set_ylabel("振幅")
    axis.set_xlim(start_sec, end_sec)
    axis.grid(alpha=0.18, linestyle="--")
    figure.tight_layout()
    return figure


def build_local_waveform_chart(
    samples: np.ndarray,
    sr: int,
    center_time: float,
    window_radius_sec: float,
    highlight_time: float | None = None,
) -> go.Figure:
    if samples.size == 0:
        figure = go.Figure()
        figure.update_layout(
            title="没有可显示的局部波形",
            template="plotly_white",
            height=240,
        )
        return figure

    start_sec = max(0.0, float(center_time - window_radius_sec))
    end_sec = min(float(samples.shape[-1] / sr), float(center_time + window_radius_sec))
    start_sample = max(0, int(start_sec * sr))
    end_sample = min(samples.shape[-1], max(start_sample + 1, int(end_sec * sr)))

    local_samples = samples[start_sample:end_sample]
    local_time = np.linspace(start_sec, end_sec, local_samples.shape[0], endpoint=False)
    if highlight_time is None:
        highlight_time = center_time

    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=local_time,
            y=local_samples,
            mode="lines",
            line={"color": "#1f5c99", "width": 1.0},
            name="波形",
            hovertemplate="时间 %{x:.3f}s<br>振幅 %{y:.4f}<extra></extra>",
        )
    )
    figure.add_vline(
        x=float(highlight_time),
        line_color="#d1495b",
        line_width=2.0,
        opacity=0.95,
    )
    figure.update_layout(
        title=f"事件附近局部波形：{center_time:.2f}s",
        template="plotly_white",
        height=260,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        xaxis={"title": "时间（秒）", "showgrid": True, "gridcolor": "rgba(0,0,0,0.08)"},
        yaxis={"title": "振幅", "showgrid": True, "gridcolor": "rgba(0,0,0,0.08)"},
        hovermode="x unified",
        dragmode="pan",
        showlegend=False,
    )
    return figure


def build_synced_waveform_player_html(
    audio_bytes: bytes,
    samples: np.ndarray,
    sr: int,
    clip_start_sec: float,
    clip_end_sec: float,
    event_time_sec: float | None = None,
    amplitude_scale: float = 1.0,
) -> str:
    if samples.size == 0:
        return """
<div style="font-family: sans-serif; color: #666; padding: 1rem;">
  没有可显示的同步试听波形
</div>
"""

    start_sample = max(0, int(clip_start_sec * sr))
    end_sample = min(samples.shape[-1], max(start_sample + 1, int(clip_end_sec * sr)))
    clip_samples = np.asarray(samples[start_sample:end_sample], dtype=np.float32)
    clip_duration = max(clip_end_sec - clip_start_sec, 1e-6)

    bucket_count = int(min(1200, max(240, clip_samples.shape[0] // 128)))
    buckets = np.array_split(np.abs(clip_samples), bucket_count)
    envelope = [round(float(np.max(bucket)) if bucket.size else 0.0, 6) for bucket in buckets]

    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    envelope_json = json.dumps(envelope, ensure_ascii=False)
    event_offset = None if event_time_sec is None else max(0.0, min(clip_duration, event_time_sec - clip_start_sec))
    event_offset_json = "null" if event_offset is None else f"{event_offset:.6f}"

    return f"""
<div style="font-family: 'Helvetica Neue', Arial, sans-serif; color: #1f1f1f; overflow: visible; padding-left: 10px; padding-right: 6px; box-sizing: border-box;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem; font-size:0.92rem;">
    <div>同步试听波形</div>
    <div id="timeLabel" style="color:#666;">{format_seconds(clip_start_sec)} / {format_seconds(clip_end_sec)}</div>
  </div>
  <audio id="audio" controls preload="auto" style="width:100%; margin-bottom:0.45rem;">
    <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav" />
  </audio>
  <canvas id="waveCanvas" width="1200" height="220" style="width:100%; height:220px; border:1px solid #e3e3e3; border-radius:8px; background:#fbfbfc; cursor:pointer;"></canvas>
  <div style="display:flex; justify-content:space-between; margin-top:0.3rem; font-size:0.82rem; color:#666;">
    <span>{format_seconds(clip_start_sec)}</span>
    <span>点击波形可跳转；绿线是播放头，红线是事件时刻</span>
    <span>{format_seconds(clip_end_sec)}</span>
  </div>
</div>
<script>
  const audio = document.getElementById("audio");
  const canvas = document.getElementById("waveCanvas");
  const ctx = canvas.getContext("2d");
  const timeLabel = document.getElementById("timeLabel");
  const envelope = {envelope_json};
  const clipDuration = {clip_duration:.6f};
  const clipStart = {clip_start_sec:.6f};
  const eventOffset = {event_offset_json};
  const amplitudeScale = {float(amplitude_scale):.3f};

  function formatTime(totalSeconds) {{
    const safe = Math.max(0, totalSeconds);
    const minutes = Math.floor(safe / 60);
    const seconds = safe - minutes * 60;
    return `${{String(minutes).padStart(2, "0")}}:${{seconds.toFixed(2).padStart(5, "0")}}`;
  }}

  function draw() {{
    const width = canvas.width;
    const height = canvas.height;
    const mid = height / 2;
    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = "#fbfbfc";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "#d8dde6";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(width, mid);
    ctx.stroke();

    const barWidth = width / envelope.length;
    ctx.fillStyle = "#1f5c99";
    for (let i = 0; i < envelope.length; i += 1) {{
      const amp = Math.max(0.01, envelope[i]);
      const barHeight = Math.min(height * 0.48, amp * (height * 0.56) * amplitudeScale);
      const x = i * barWidth;
      ctx.fillRect(x, mid - barHeight, Math.max(1, barWidth * 0.88), barHeight * 2);
    }}

    if (eventOffset !== null && clipDuration > 0) {{
      const eventX = (eventOffset / clipDuration) * width;
      ctx.strokeStyle = "#d1495b";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(eventX, 0);
      ctx.lineTo(eventX, height);
      ctx.stroke();
    }}

    const currentX = clipDuration > 0 ? (audio.currentTime / clipDuration) * width : 0;
    ctx.strokeStyle = "#17c964";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(currentX, 0);
    ctx.lineTo(currentX, height);
    ctx.stroke();

    const absoluteCurrent = clipStart + (audio.currentTime || 0);
    timeLabel.textContent = `${{formatTime(absoluteCurrent)}} / {format_seconds(clip_end_sec)}`;
  }}

  function tick() {{
    draw();
    if (!audio.paused && !audio.ended) {{
      window.requestAnimationFrame(tick);
    }}
  }}

  ["play", "pause", "seeked", "loadedmetadata", "timeupdate", "ended"].forEach((eventName) => {{
    audio.addEventListener(eventName, () => {{
      draw();
      if (eventName === "play") {{
        window.requestAnimationFrame(tick);
      }}
    }});
  }});

  canvas.addEventListener("click", (event) => {{
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    audio.currentTime = ratio * clipDuration;
    draw();
  }});

  draw();
</script>
"""


def _spectrogram_png_base64(
    spectrogram_db: np.ndarray,
    spectrogram_freqs: np.ndarray | None = None,
    frequency_scale: str = "linear",
) -> str:
    if spectrogram_db.size == 0:
        empty = Image.new("RGBA", (2, 2), color=(255, 255, 255, 255))
        buffer = io.BytesIO()
        empty.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    spectrum = np.asarray(spectrogram_db, dtype=np.float32)
    if (
        frequency_scale == "log"
        and spectrogram_freqs is not None
        and len(spectrogram_freqs) == spectrum.shape[0]
    ):
        frequencies = np.asarray(spectrogram_freqs, dtype=np.float64)
        positive = frequencies >= 20.0
        if np.any(positive):
            source_freqs = frequencies[positive]
            source_spectrum = spectrum[positive, :]
            target_freqs = np.geomspace(
                source_freqs[0],
                source_freqs[-1],
                num=source_spectrum.shape[0],
            )
            source_indices = np.searchsorted(source_freqs, target_freqs, side="left")
            source_indices = np.clip(source_indices, 0, source_freqs.size - 1)
            previous_indices = np.maximum(source_indices - 1, 0)
            choose_previous = (
                np.abs(source_freqs[previous_indices] - target_freqs)
                <= np.abs(source_freqs[source_indices] - target_freqs)
            )
            source_indices = np.where(choose_previous, previous_indices, source_indices)
            spectrum = source_spectrum[source_indices, :]
    floor_db = float(np.clip(np.min(spectrum), -120.0, -20.0))
    clipped = np.clip(spectrum, floor_db, 0.0)
    normalized = (clipped - floor_db) / max(-floor_db, 1e-8)
    enhanced = np.power(np.clip(normalized, 0.0, 1.0), 0.72)
    rgba = (_SPECTROGRAM_CMAP(enhanced) * 255).astype(np.uint8)
    rgba = np.flipud(rgba)
    image = Image.fromarray(rgba)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def figure_to_png_bytes(figure: plt.Figure, dpi: int = 220) -> bytes:
    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)
    return buffer.getvalue()


def build_synced_overview_player_html(
    audio_bytes: bytes,
    audio_mime: str,
    samples: np.ndarray,
    sr: int,
    spectrogram_db: np.ndarray,
    spectrogram_times: np.ndarray,
    spectrogram_freqs: np.ndarray,
    duration_sec: float,
    initial_wave_amplitude_max: float | None = None,
    initial_freq_max_hz: float | None = None,
    frequency_scale: str = "linear",
) -> str:
    if samples.size == 0:
        return """
<div style="font-family: sans-serif; color: #666; padding: 1rem;">
  没有可显示的同步总览
</div>
"""

    sample_count = samples.shape[-1]
    duration_sec = max(float(duration_sec), 1e-6)
    bucket_count = int(min(2200, max(480, sample_count // 256)))
    buckets = np.array_split(np.abs(np.asarray(samples, dtype=np.float32)), bucket_count)
    envelope = [round(float(np.max(bucket)) if bucket.size else 0.0, 6) for bucket in buckets]
    envelope_json = json.dumps(envelope, ensure_ascii=False)
    if initial_wave_amplitude_max is None:
        envelope_array = np.asarray(envelope, dtype=np.float32)
        if envelope_array.size:
            auto_wave_amplitude_max = float(np.quantile(envelope_array, 0.98) * 1.08)
        else:
            auto_wave_amplitude_max = 0.3
        visible_wave_amplitude_max = max(0.03, min(1.0, auto_wave_amplitude_max))
    else:
        visible_wave_amplitude_max = max(0.03, min(1.0, float(initial_wave_amplitude_max)))

    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    resolved_frequency_scale = "log" if frequency_scale == "log" else "linear"
    spectrogram_b64 = _spectrogram_png_base64(
        spectrogram_db,
        spectrogram_freqs,
        resolved_frequency_scale,
    )
    full_freq_max = float(spectrogram_freqs[-1]) if spectrogram_freqs.size else 22050.0
    positive_freqs = spectrogram_freqs[spectrogram_freqs >= 20.0]
    minimum_display_freq = float(positive_freqs[0]) if positive_freqs.size else 20.0
    visible_freq_max = full_freq_max
    if initial_freq_max_hz is not None:
        visible_freq_max = max(1000.0, min(float(initial_freq_max_hz), full_freq_max))
    freq_slider_max = max(1000.0, float(int(np.ceil(full_freq_max / 500.0) * 500.0)))

    spec_time_start = float(spectrogram_times[0]) if spectrogram_times.size else 0.0
    spec_time_end = float(spectrogram_times[-1]) if spectrogram_times.size else duration_sec
    spec_time_span = max(spec_time_end - spec_time_start, 1e-6)
    window_slider_max = max(5.0, float(int(np.ceil(duration_sec))))
    initial_window_sec = window_slider_max
    default_focus_window_sec = min(
        duration_sec,
        max(15.0, min(30.0, duration_sec / 6.0 if duration_sec > 60 else duration_sec / 2.0)),
    )

    return f"""
<div style="font-family: 'Helvetica Neue', Arial, sans-serif; color: #1f1f1f;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem; font-size:0.95rem;">
    <div>同步总览：纯频谱 + 波形</div>
    <div id="overviewTimeLabel" style="color:#666;">00:00.00 / {format_seconds(duration_sec)}</div>
  </div>
  <audio id="overviewAudio" controls preload="auto" style="width:100%; margin-bottom:0.6rem;">
    <source src="data:{audio_mime};base64,{audio_b64}" type="{audio_mime}" />
  </audio>
  <div style="display:grid; grid-template-columns: 1fr; gap: 0.8rem; margin-bottom:0.6rem; font-size:0.88rem;">
    <div style="display:flex; justify-content:flex-start; margin-bottom:0.1rem;">
      <button
        id="overviewViewModeToggle"
        type="button"
        style="border:1px solid #b8c4d3; background:#f6f9fc; color:#243447; border-radius:999px; padding:0.35rem 0.8rem; font-size:0.82rem; cursor:pointer;"
      >
        切换到局部聚焦
      </button>
    </div>
    <label style="display:flex; flex-direction:column; gap:0.2rem;">
      <span>聚焦窗口：<span id="overviewWindowValue">{initial_window_sec:.0f}</span> 秒</span>
      <input id="overviewWindowSec" type="range" min="5" max="{int(window_slider_max)}" step="1" value="{initial_window_sec:.0f}" />
    </label>
  </div>
  <div id="overviewWindowLabel" style="margin-bottom:0.45rem; font-size:0.84rem; color:#5c6773;">
    当前聚焦区间：00:00.00 - {format_seconds(min(duration_sec, initial_window_sec))}
  </div>
  <div style="display:flex; align-items:stretch; gap:0.8rem; margin-bottom:0.55rem; overflow:visible;">
    <div style="flex:1 1 auto; min-width:0; display:flex; flex-direction:column; gap:0.35rem; overflow:visible;">
      <canvas id="overviewWaveCanvas" width="1400" height="230" style="flex:1; width:100%; height:230px; border:1px solid #e3e3e3; border-radius:8px; background:#fbfbfc; cursor:crosshair; display:block;"></canvas>
      <div id="overviewWaveHover" style="font-size:0.82rem; color:#5c6773;">悬停波形：时间 -- | 包络振幅 --</div>
    </div>
    <div id="overviewWaveControl" style="flex:0 0 72px; width:72px; border:1px solid #d9e0e8; border-radius:10px; background:#f7f9fc; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:0.5rem 0.35rem; gap:0.45rem;">
      <div style="font-size:0.78rem; color:#4f5d75; text-align:center;">波形纵轴上限</div>
      <div id="overviewWaveAmplitudeValue" style="font-size:0.92rem; font-weight:600; color:#1f2d3d;">±{float(visible_wave_amplitude_max):.2f}</div>
      <input
        id="overviewWaveAmplitudeMax"
        type="range"
        min="0.10"
        max="1.00"
        step="0.05"
        value="{float(visible_wave_amplitude_max):.2f}"
        orient="vertical"
        style="-webkit-appearance: slider-vertical; writing-mode: vertical-lr; direction: rtl; width: 26px; height: 150px;"
      />
    </div>
  </div>
  <div style="display:flex; align-items:stretch; gap:0.8rem; overflow:visible;">
    <div style="flex:1 1 auto; min-width:0; display:flex; flex-direction:column; gap:0.35rem; overflow:visible;">
      <canvas id="overviewSpecCanvas" width="1400" height="360" style="flex:1; width:100%; height:360px; border:1px solid #e3e3e3; border-radius:8px; background:#111; cursor:crosshair; display:block;"></canvas>
      <div id="overviewSpecHover" style="font-size:0.82rem; color:#5c6773;">悬停频谱：时间 -- | 频率 --</div>
      <div style="display:flex; align-items:center; gap:0.5rem; font-size:0.76rem; color:#5c6773;">
        <span>弱</span>
        <div style="height:8px; flex:0 1 260px; border-radius:999px; background:linear-gradient(90deg,#020308 0%,#211446 28%,#1652a3 55%,#21d4e8 82%,#f4ff72 100%);"></div>
        <span>强</span>
      </div>
    </div>
    <div id="overviewFreqControl" style="flex:0 0 84px; width:84px; border:1px solid #2a2f39; border-radius:10px; background:#171b22; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:0.5rem 0.35rem; gap:0.45rem;">
      <div style="font-size:0.78rem; color:#d7dde8; text-align:center;">频率上限</div>
      <div id="overviewFreqValue" style="font-size:0.92rem; font-weight:600; color:#ffffff;">{int(visible_freq_max)} Hz</div>
      <input
        id="overviewFreqMax"
        type="range"
        min="1000"
        max="{int(freq_slider_max)}"
        step="250"
        value="{int(visible_freq_max)}"
        orient="vertical"
        style="-webkit-appearance: slider-vertical; writing-mode: vertical-lr; direction: rtl; width: 26px; height: 230px;"
      />
    </div>
  </div>
  <div style="display:flex; justify-content:space-between; margin-top:0.35rem; font-size:0.82rem; color:#666;">
    <span id="overviewWindowStart">00:00.00</span>
    <span>滚轮缩放时间范围，拖拽平移窗口，双击回到播放头；上面的进度条拖到哪里，下面就聚焦哪里</span>
    <span id="overviewWindowEnd">{format_seconds(min(duration_sec, initial_window_sec))}</span>
  </div>
</div>
<script>
  const audio = document.getElementById("overviewAudio");
  const waveCanvas = document.getElementById("overviewWaveCanvas");
  const specCanvas = document.getElementById("overviewSpecCanvas");
  const waveCtx = waveCanvas.getContext("2d");
  const specCtx = specCanvas.getContext("2d");
  const timeLabel = document.getElementById("overviewTimeLabel");
  const viewModeToggle = document.getElementById("overviewViewModeToggle");
  const windowInput = document.getElementById("overviewWindowSec");
  const waveAmplitudeInput = document.getElementById("overviewWaveAmplitudeMax");
  const freqInput = document.getElementById("overviewFreqMax");
  const windowValue = document.getElementById("overviewWindowValue");
  const waveAmplitudeValue = document.getElementById("overviewWaveAmplitudeValue");
  const freqValue = document.getElementById("overviewFreqValue");
  const waveHoverLabel = document.getElementById("overviewWaveHover");
  const specHoverLabel = document.getElementById("overviewSpecHover");
  const windowLabel = document.getElementById("overviewWindowLabel");
  const windowStartLabel = document.getElementById("overviewWindowStart");
  const windowEndLabel = document.getElementById("overviewWindowEnd");
  const envelope = {envelope_json};
  const duration = {duration_sec:.6f};
  const specTimeStart = {spec_time_start:.6f};
  const specTimeSpan = {spec_time_span:.6f};
  const fullFreqMax = {full_freq_max:.3f};
  const minimumDisplayFreq = {minimum_display_freq:.3f};
  const frequencyScale = "{resolved_frequency_scale}";
  const spectrogramImage = new Image();
  spectrogramImage.src = "data:image/png;base64,{spectrogram_b64}";
  const waveAxisPad = 96;
  const specAxisPad = 120;
  const rightPad = 12;
  const waveTopPad = 16;
  const waveBottomPad = 16;
  const specTopPad = 18;
  const specBottomPad = 22;
  const minViewSpan = Math.min(duration, 1.0);
  let viewSpan = Math.min(duration, {initial_window_sec:.6f});
  let viewStart = 0;
  let followPlayhead = true;
  let dragState = null;
  let lastFocusSpan = Math.max(minViewSpan, Math.min(duration, {default_focus_window_sec:.6f}));

  function formatTime(totalSeconds) {{
    const safe = Math.max(0, totalSeconds);
    const minutes = Math.floor(safe / 60);
    const seconds = safe - minutes * 60;
    return `${{String(minutes).padStart(2, "0")}}:${{seconds.toFixed(2).padStart(5, "0")}}`;
  }}

  function clampView(start, span) {{
    const safeSpan = Math.max(minViewSpan, Math.min(duration, span));
    const safeStart = Math.max(0, Math.min(duration - safeSpan, start));
    return {{
      start: safeStart,
      end: safeStart + safeSpan,
      span: safeSpan,
    }};
  }}

  function centeredView(center, span) {{
    return clampView(center - span / 2, span);
  }}

  function currentWindowBounds() {{
    return clampView(viewStart, viewSpan);
  }}

  function isFullView(windowBounds) {{
    return windowBounds.start <= 1e-3 && Math.abs(windowBounds.span - duration) <= 0.5;
  }}

  function syncViewToPlayhead() {{
    const centered = centeredView(audio.currentTime || 0, viewSpan);
    viewStart = centered.start;
    viewSpan = centered.span;
  }}

  function playheadX(currentTime, left, plotWidth, windowStart, windowSpan) {{
    const relative = windowSpan > 0 ? (currentTime - windowStart) / windowSpan : 0;
    return left + Math.max(0, Math.min(1, relative)) * plotWidth;
  }}

  function canvasMetrics(canvasEl) {{
    const rect = canvasEl.getBoundingClientRect();
    const internalWidth = canvasEl.width;
    const leftPadInternal = canvasEl === specCanvas ? specAxisPad : waveAxisPad;
    const rightPadInternal = rightPad;
    const leftPadDisplay = (leftPadInternal / internalWidth) * rect.width;
    const rightPadDisplay = (rightPadInternal / internalWidth) * rect.width;
    return {{
      rect,
      leftPadDisplay,
      rightPadDisplay,
      usableDisplayWidth: Math.max(1, rect.width - leftPadDisplay - rightPadDisplay),
    }};
  }}

  function eventRatioWithinPlot(event, canvasEl) {{
    const metrics = canvasMetrics(canvasEl);
    const localX = event.clientX - metrics.rect.left;
    const clampedX = Math.max(
      metrics.leftPadDisplay,
      Math.min(metrics.rect.width - metrics.rightPadDisplay, localX),
    );
    return Math.max(0, Math.min(1, (clampedX - metrics.leftPadDisplay) / metrics.usableDisplayWidth));
  }}

  function eventYRatioWithinCanvas(event, canvasEl) {{
    const rect = canvasEl.getBoundingClientRect();
    const topPad = canvasEl === specCanvas ? specTopPad : waveTopPad;
    const bottomPad = canvasEl === specCanvas ? specBottomPad : waveBottomPad;
    const internalHeight = canvasEl.height;
    const topPadDisplay = (topPad / internalHeight) * rect.height;
    const bottomPadDisplay = (bottomPad / internalHeight) * rect.height;
    const usableDisplayHeight = Math.max(1, rect.height - topPadDisplay - bottomPadDisplay);
    const localY = event.clientY - rect.top;
    const clampedY = Math.max(topPadDisplay, Math.min(rect.height - bottomPadDisplay, localY));
    return Math.max(0, Math.min(1, (clampedY - topPadDisplay) / usableDisplayHeight));
  }}

  function getVisibleEnvelope(windowBounds) {{
    const startIndex = Math.max(0, Math.floor((windowBounds.start / duration) * envelope.length));
    const endIndex = Math.min(envelope.length, Math.ceil((windowBounds.end / duration) * envelope.length));
    return envelope.slice(startIndex, Math.max(startIndex + 1, endIndex));
  }}

  function applyWindowFromSlider() {{
    const requested = parseFloat(windowInput.value || "{initial_window_sec:.0f}");
    const safeSpan = Math.max(minViewSpan, Math.min(duration, requested));
    const center = followPlayhead ? (audio.currentTime || 0) : (viewStart + viewSpan / 2);
    const next = centeredView(center, safeSpan);
    viewStart = next.start;
    viewSpan = next.span;
  }}

  function zoomAroundPointer(event, canvasEl) {{
    event.preventDefault();
    const ratio = eventRatioWithinPlot(event, canvasEl);
    const zoomFactor = event.deltaY < 0 ? 0.84 : 1.18;
    const anchorTime = viewStart + ratio * viewSpan;
    const nextSpan = Math.max(minViewSpan, Math.min(duration, viewSpan * zoomFactor));
    const nextStart = anchorTime - ratio * nextSpan;
    const next = clampView(nextStart, nextSpan);
    viewStart = next.start;
    viewSpan = next.span;
    followPlayhead = false;
    drawAll();
  }}

  function pointerDown(event, canvasEl) {{
    dragState = {{
      canvasEl,
      pointerId: event.pointerId,
      startClientX: event.clientX,
      startViewStart: viewStart,
      moved: false,
    }};
    canvasEl.setPointerCapture(event.pointerId);
  }}

  function pointerMove(event) {{
    if (!dragState || dragState.pointerId !== event.pointerId) {{
      return;
    }}
    const metrics = canvasMetrics(dragState.canvasEl);
    const deltaX = event.clientX - dragState.startClientX;
    if (Math.abs(deltaX) > 3) {{
      dragState.moved = true;
    }}
    if (!dragState.moved) {{
      return;
    }}
    const deltaSeconds = (deltaX / metrics.usableDisplayWidth) * viewSpan;
    const next = clampView(dragState.startViewStart - deltaSeconds, viewSpan);
    viewStart = next.start;
    viewSpan = next.span;
    followPlayhead = false;
    drawAll();
  }}

  function pointerUp(event) {{
    if (!dragState || dragState.pointerId !== event.pointerId) {{
      return;
    }}
    const canvasEl = dragState.canvasEl;
    const wasMoved = dragState.moved;
    dragState = null;
    canvasEl.releasePointerCapture(event.pointerId);
    if (!wasMoved) {{
      const ratio = eventRatioWithinPlot(event, canvasEl);
      const windowBounds = currentWindowBounds();
      audio.currentTime = windowBounds.start + ratio * windowBounds.span;
      if (followPlayhead) {{
        syncViewToPlayhead();
      }}
      drawAll();
    }}
  }}

  function resetToPlayhead() {{
    followPlayhead = true;
    applyWindowFromSlider();
    syncViewToPlayhead();
    drawAll();
  }}

  function toggleOverviewMode() {{
    const windowBounds = currentWindowBounds();
    if (isFullView(windowBounds)) {{
      followPlayhead = true;
      viewSpan = lastFocusSpan;
      if (windowInput) {{
        windowInput.value = `${{Math.round(viewSpan)}}`;
      }}
      syncViewToPlayhead();
    }} else {{
      lastFocusSpan = windowBounds.span;
      followPlayhead = false;
      viewStart = 0;
      viewSpan = duration;
      if (windowInput) {{
        windowInput.value = `${{Math.round(duration)}}`;
      }}
    }}
    drawAll();
  }}

  function drawWaveform() {{
    const width = waveCanvas.width;
    const height = waveCanvas.height;
    const plotTop = waveTopPad;
    const plotBottom = height - waveBottomPad;
    const plotHeight = Math.max(1, plotBottom - plotTop);
    const mid = plotTop + plotHeight / 2;
    const amplitudeMax = Math.max(0.05, parseFloat(waveAmplitudeInput.value));
    const plotLeft = waveAxisPad;
    const plotRight = width - rightPad;
    const plotWidth = Math.max(1, plotRight - plotLeft);
    const windowBounds = currentWindowBounds();
    const visibleEnvelope = getVisibleEnvelope(windowBounds);
    waveCtx.clearRect(0, 0, width, height);
    waveCtx.fillStyle = "#fbfbfc";
    waveCtx.fillRect(0, 0, width, height);

    waveCtx.fillStyle = "#4f5d75";
    waveCtx.font = "13px sans-serif";
    waveCtx.textAlign = "right";
    waveCtx.textBaseline = "middle";
    const waveTicks = [amplitudeMax, amplitudeMax / 2, 0.0, -amplitudeMax / 2, -amplitudeMax];
    for (const tick of waveTicks) {{
      const normalized = amplitudeMax > 0 ? tick / amplitudeMax : 0;
      const y = mid - normalized * (plotHeight * 0.42);
      waveCtx.fillText(tick > 0 ? `+${{tick.toFixed(2)}}` : tick.toFixed(2), plotLeft - 12, y);
      waveCtx.strokeStyle = "rgba(79, 93, 117, 0.18)";
      waveCtx.lineWidth = 1;
      waveCtx.beginPath();
      waveCtx.moveTo(plotLeft - 4, y);
      waveCtx.lineTo(plotRight, y);
      waveCtx.stroke();
    }}
    waveCtx.save();
    waveCtx.translate(24, plotTop + plotHeight / 2);
    waveCtx.rotate(-Math.PI / 2);
    waveCtx.fillStyle = "#4f5d75";
    waveCtx.textAlign = "center";
    waveCtx.fillText("振幅", 0, 0);
    waveCtx.restore();

    waveCtx.strokeStyle = "#9aa5b1";
    waveCtx.lineWidth = 1;
    waveCtx.beginPath();
    waveCtx.moveTo(plotLeft, plotTop);
    waveCtx.lineTo(plotLeft, plotBottom);
    waveCtx.moveTo(plotLeft, mid);
    waveCtx.lineTo(plotRight, mid);
    waveCtx.stroke();

    const barWidth = plotWidth / visibleEnvelope.length;
    waveCtx.fillStyle = "#1f5c99";
    for (let i = 0; i < visibleEnvelope.length; i += 1) {{
      const amp = Math.max(0.01, visibleEnvelope[i]);
      const normalizedAmp = Math.min(1.0, amp / amplitudeMax);
      const barHeight = normalizedAmp * (plotHeight * 0.42);
      const x = plotLeft + i * barWidth;
      waveCtx.fillRect(x, mid - barHeight, Math.max(1, barWidth * 0.86), barHeight * 2);
    }}

    const x = playheadX(audio.currentTime, plotLeft, plotWidth, windowBounds.start, windowBounds.span);
    waveCtx.strokeStyle = "#17c964";
    waveCtx.lineWidth = 2.5;
    waveCtx.beginPath();
    waveCtx.moveTo(x, plotTop);
    waveCtx.lineTo(x, plotBottom);
    waveCtx.stroke();
  }}

  function drawSpectrogram() {{
    const width = specCanvas.width;
    const height = specCanvas.height;
    const plotTop = specTopPad;
    const plotBottom = height - specBottomPad;
    const plotHeight = Math.max(1, plotBottom - plotTop);
    const freqMax = parseFloat(freqInput.value);
    const plotLeft = specAxisPad;
    const plotRight = width - rightPad;
    const plotWidth = Math.max(1, plotRight - plotLeft);
    const windowBounds = currentWindowBounds();
    specCtx.clearRect(0, 0, width, height);
    specCtx.fillStyle = "#111";
    specCtx.fillRect(0, 0, width, height);

    if (spectrogramImage.complete) {{
      const frequencyRatio = frequencyScale === "log"
        ? Math.log(Math.max(freqMax, minimumDisplayFreq) / minimumDisplayFreq)
          / Math.log(fullFreqMax / minimumDisplayFreq)
        : freqMax / fullFreqMax;
      const sourceHeight = spectrogramImage.height * Math.max(0, Math.min(1, frequencyRatio));
      const sourceY = spectrogramImage.height - sourceHeight;
      const timeStartRatio = specTimeSpan > 0 ? (windowBounds.start - specTimeStart) / specTimeSpan : 0;
      const timeEndRatio = specTimeSpan > 0 ? (windowBounds.end - specTimeStart) / specTimeSpan : 1;
      const sourceX = Math.max(0, Math.min(spectrogramImage.width - 1, timeStartRatio * spectrogramImage.width));
      const sourceX2 = Math.max(sourceX + 1, Math.min(spectrogramImage.width, timeEndRatio * spectrogramImage.width));
      specCtx.drawImage(
        spectrogramImage,
        sourceX,
        sourceY,
        sourceX2 - sourceX,
        sourceHeight,
        plotLeft,
        plotTop,
        plotWidth,
        plotHeight,
      );
    }}

    specCtx.strokeStyle = "rgba(255,255,255,0.22)";
    specCtx.lineWidth = 1;
    specCtx.beginPath();
    specCtx.moveTo(plotLeft, plotTop);
    specCtx.lineTo(plotLeft, plotBottom);
    specCtx.stroke();

    specCtx.fillStyle = "#ffffff";
    specCtx.font = "13px sans-serif";
    specCtx.textAlign = "right";
    specCtx.textBaseline = "middle";
    const freqTicks = 5;
    for (let i = 0; i < freqTicks; i += 1) {{
      const ratio = i / (freqTicks - 1);
      const y = plotBottom - ratio * plotHeight;
      const tickFreq = frequencyScale === "log"
        ? minimumDisplayFreq * Math.pow(freqMax / minimumDisplayFreq, ratio)
        : ratio * freqMax;
      specCtx.fillText(`${{Math.round(tickFreq)}}`, plotLeft - 12, y);
      specCtx.strokeStyle = "rgba(255,255,255,0.14)";
      specCtx.lineWidth = 1;
      specCtx.beginPath();
      specCtx.moveTo(plotLeft - 4, y);
      specCtx.lineTo(plotRight, y);
      specCtx.stroke();
    }}
    specCtx.save();
    specCtx.translate(28, plotTop + plotHeight / 2);
    specCtx.rotate(-Math.PI / 2);
    specCtx.fillStyle = "#ffffff";
    specCtx.textAlign = "center";
    specCtx.fillText("频率 ({'Hz，对数' if resolved_frequency_scale == 'log' else 'Hz'})", 0, 0);
    specCtx.restore();

    const x = playheadX(audio.currentTime, plotLeft, plotWidth, windowBounds.start, windowBounds.span);
    specCtx.strokeStyle = "#17c964";
    specCtx.lineWidth = 2.5;
    specCtx.beginPath();
    specCtx.moveTo(x, plotTop);
    specCtx.lineTo(x, plotBottom);
    specCtx.stroke();
  }}

  function updateWaveHover(event) {{
    const ratio = eventRatioWithinPlot(event, waveCanvas);
    const windowBounds = currentWindowBounds();
    const hoverTime = windowBounds.start + ratio * windowBounds.span;
    const visibleEnvelope = getVisibleEnvelope(windowBounds);
    const index = Math.min(visibleEnvelope.length - 1, Math.max(0, Math.floor(ratio * visibleEnvelope.length)));
    const amplitude = visibleEnvelope[index] || 0;
    waveHoverLabel.textContent = `悬停波形：时间 ${{formatTime(hoverTime)}} | 包络振幅 ${{amplitude.toFixed(4)}}`;
  }}

  function updateSpecHover(event) {{
    const ratio = eventRatioWithinPlot(event, specCanvas);
    const yRatio = eventYRatioWithinCanvas(event, specCanvas);
    const windowBounds = currentWindowBounds();
    const hoverTime = windowBounds.start + ratio * windowBounds.span;
    const freqMax = parseFloat(freqInput.value);
    const verticalRatio = 1 - yRatio;
    const hoverFreq = frequencyScale === "log"
      ? minimumDisplayFreq * Math.pow(freqMax / minimumDisplayFreq, verticalRatio)
      : verticalRatio * freqMax;
    specHoverLabel.textContent = `悬停频谱：时间 ${{formatTime(hoverTime)}} | 频率 ${{hoverFreq.toFixed(1)}} Hz`;
  }}

  function resetHoverLabels() {{
    waveHoverLabel.textContent = "悬停波形：时间 -- | 包络振幅 --";
    specHoverLabel.textContent = "悬停频谱：时间 -- | 频率 --";
  }}

  function drawAll() {{
    if (followPlayhead) {{
      syncViewToPlayhead();
    }}
    const windowBounds = currentWindowBounds();
    if (!isFullView(windowBounds)) {{
      lastFocusSpan = windowBounds.span;
    }}
    windowInput.value = `${{Math.round(windowBounds.span)}}`;
    windowValue.textContent = `${{Math.round(windowBounds.span)}}`;
    waveAmplitudeValue.textContent = `±${{parseFloat(waveAmplitudeInput.value).toFixed(2)}}`;
    freqValue.textContent = `${{Math.round(parseFloat(freqInput.value))}}`;
    viewModeToggle.textContent = isFullView(windowBounds) ? "切换到局部聚焦" : "切换到全曲总览";
    windowLabel.textContent = `当前聚焦区间：${{formatTime(windowBounds.start)}} - ${{formatTime(windowBounds.end)}}`;
    windowStartLabel.textContent = formatTime(windowBounds.start);
    windowEndLabel.textContent = formatTime(windowBounds.end);
    drawWaveform();
    drawSpectrogram();
    timeLabel.textContent = `${{formatTime(audio.currentTime || 0)}} / {format_seconds(duration_sec)}`;
  }}

  function tick() {{
    drawAll();
    if (!audio.paused && !audio.ended) {{
      window.requestAnimationFrame(tick);
    }}
  }}

  ["play", "pause", "seeked", "loadedmetadata", "timeupdate", "ended"].forEach((eventName) => {{
    audio.addEventListener(eventName, () => {{
      if (eventName === "seeked" || eventName === "loadedmetadata") {{
        if (followPlayhead) {{
          syncViewToPlayhead();
        }} else {{
          const next = centeredView(audio.currentTime || 0, viewSpan);
          viewStart = next.start;
          viewSpan = next.span;
        }}
      }}
      drawAll();
      if (eventName === "play") {{
        window.requestAnimationFrame(tick);
      }}
    }});
  }});

  windowInput.addEventListener("input", () => {{
    followPlayhead = false;
    applyWindowFromSlider();
    drawAll();
  }});
  viewModeToggle.addEventListener("click", toggleOverviewMode);
  waveAmplitudeInput.addEventListener("input", drawAll);
  freqInput.addEventListener("input", drawAll);
  [waveCanvas, specCanvas].forEach((canvasEl) => {{
    canvasEl.addEventListener("wheel", (event) => zoomAroundPointer(event, canvasEl), {{ passive: false }});
    canvasEl.addEventListener("pointerdown", (event) => pointerDown(event, canvasEl));
    canvasEl.addEventListener("pointermove", pointerMove);
    canvasEl.addEventListener("pointerup", pointerUp);
    canvasEl.addEventListener("pointercancel", pointerUp);
    canvasEl.addEventListener("dblclick", resetToPlayhead);
  }});
  waveCanvas.addEventListener("mousemove", updateWaveHover);
  specCanvas.addEventListener("mousemove", updateSpecHover);
  waveCanvas.addEventListener("mouseleave", resetHoverLabels);
  specCanvas.addEventListener("mouseleave", resetHoverLabels);
  syncViewToPlayhead();
  spectrogramImage.onload = drawAll;
  resetHoverLabels();
  drawAll();
</script>
"""


def plot_local_spectrogram(
    spectrogram_db: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    center_time: float,
    window_radius_sec: float,
    highlight_time: float | None = None,
    max_freq_hz: float | None = 10000.0,
    samples: np.ndarray | None = None,
    sr: int | None = None,
    n_fft: int = 4096,
    hop_length: int = 1024,
    dynamic_range_db: float = 90.0,
    frequency_scale: str = "linear",
) -> plt.Figure:
    local_source_note = ""
    if samples is not None and sr is not None and np.asarray(samples).size >= 2:
        audio_samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        clip_start = max(0.0, float(center_time - window_radius_sec))
        clip_end = min(float(audio_samples.size / sr), float(center_time + window_radius_sec))
        clip = audio_samples[int(clip_start * sr) : max(int(clip_start * sr) + 2, int(clip_end * sr))]
        if clip.size >= 2:
            local_time_parts: list[np.ndarray] = []
            local_magnitude_parts: list[np.ndarray] = []
            local_freqs = np.array([], dtype=np.float32)
            for block_freqs, block_times, magnitude in _iter_magnitude_blocks(
                clip,
                int(sr),
                int(n_fft),
                int(hop_length),
            ):
                local_freqs = block_freqs
                local_time_parts.append(block_times + clip_start)
                local_magnitude_parts.append(magnitude)
            if local_time_parts and local_magnitude_parts:
                times = np.concatenate(local_time_parts).astype(np.float32)
                freqs = local_freqs
                local_magnitude = np.concatenate(local_magnitude_parts, axis=1)
                spectrogram_db = _magnitude_to_relative_db(
                    local_magnitude,
                    dynamic_range_db,
                )
                local_source_note = "（局部峰值 = 0 dB）"

    if times.size == 0 or freqs.size == 0 or spectrogram_db.size == 0:
        figure, axis = plt.subplots(figsize=(8.2, 3.8))
        axis.text(0.5, 0.5, "没有可显示的局部频谱", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        figure.tight_layout()
        return figure

    start_time = max(float(times[0]), float(center_time - window_radius_sec))
    end_time = min(float(times[-1]), float(center_time + window_radius_sec))
    time_mask = (times >= start_time) & (times <= end_time)
    if not np.any(time_mask):
        nearest_time = int(np.argmin(np.abs(times - center_time)))
        time_mask[max(0, nearest_time - 2) : min(times.size, nearest_time + 3)] = True

    freq_mask = np.ones_like(freqs, dtype=bool)
    if max_freq_hz is not None:
        freq_mask = freqs <= float(max_freq_hz)
        if not np.any(freq_mask):
            freq_mask[:] = True

    local_times = times[time_mask]
    local_freqs = freqs[freq_mask]
    local_db = spectrogram_db[np.ix_(freq_mask, time_mask)]
    local_db, local_freqs = _prepare_frequency_display(
        local_db,
        local_freqs,
        frequency_scale,
    )

    figure, axis = plt.subplots(figsize=(8.8, 4.2))
    mesh = axis.pcolormesh(
        local_times,
        local_freqs / 1000.0,
        local_db,
        shading="auto",
        cmap=_SPECTROGRAM_CMAP,
        norm=_spectrogram_norm(dynamic_range_db),
        rasterized=True,
    )

    if highlight_time is None:
        highlight_time = center_time
    axis.axvline(float(highlight_time), color="#ffffff", alpha=0.95, linewidth=1.2)
    axis.set_title(f"事件附近局部频谱：{center_time:.2f}s {local_source_note}")
    axis.set_xlabel("时间（秒）")
    _configure_frequency_axis(axis, local_freqs, frequency_scale)
    axis.grid(axis="y", color="white", alpha=0.10, linewidth=0.5)
    axis.set_xlim(float(local_times[0]), float(local_times[-1]))
    colorbar = figure.colorbar(mesh, ax=axis, format="%+2.0f dB", pad=0.02, fraction=0.04)
    colorbar.set_label("相对幅度（dB）")
    figure.subplots_adjust(left=0.15, right=0.90, bottom=0.12, top=0.90)
    return figure


def plot_event_density(
    event_times: np.ndarray,
    duration_sec: float,
    major_event_mask: np.ndarray | list[bool] | None = None,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(14, 2.8))
    times = np.asarray(event_times, dtype=np.float64)
    if times.size == 0:
        axis.text(0.5, 0.5, "未检测到事件", ha="center", va="center", transform=axis.transAxes)
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.set_xlim(0, max(duration_sec, 1.0))
        axis.set_title("事件密度分布")
        _format_time_axis(axis, duration_sec)
        figure.tight_layout()
        return figure

    safe_duration = max(float(duration_sec), 1e-6)
    bin_count = min(48, max(6, int(np.ceil(safe_duration / 30.0))))
    counts, edges = np.histogram(times, bins=bin_count, range=(0.0, safe_duration))
    bin_width_sec = float(edges[1] - edges[0])
    rates_per_minute = counts / max(bin_width_sec / 60.0, 1e-8)
    centers = (edges[:-1] + edges[1:]) / 2.0
    axis.bar(
        centers,
        rates_per_minute,
        width=bin_width_sec * 0.92,
        color="#457b9d",
        alpha=0.82,
        linewidth=0.0,
    )
    major_flags = (
        np.asarray(major_event_mask, dtype=bool)
        if major_event_mask is not None
        else np.zeros(times.size, dtype=bool)
    )
    if major_flags.size != times.size:
        major_flags = np.zeros(times.size, dtype=bool)
    ordinary_times = times[~major_flags]
    major_times = times[major_flags]
    marker_height = max(float(np.max(rates_per_minute)) * 0.04, 0.1)
    if ordinary_times.size:
        axis.scatter(
            ordinary_times,
            np.full(ordinary_times.shape, marker_height),
            marker="|",
            s=90,
            linewidths=1.0,
            color="#f4a261",
            label="普通事件",
            zorder=3,
        )
    if major_times.size:
        axis.scatter(
            major_times,
            np.full(major_times.shape, marker_height),
            marker="|",
            s=130,
            linewidths=2.0,
            color="#c44536",
            label="候选边界",
            zorder=4,
        )
    axis.set_title("事件密度与候选边界")
    _format_time_axis(axis, duration_sec)
    axis.set_ylabel("事件率（个/分钟）")
    axis.set_xlim(0, max(duration_sec, 1.0))
    axis.grid(alpha=0.2, linestyle="--")
    if major_times.size:
        axis.legend(loc="upper right", frameon=False, ncols=2)
    figure.tight_layout()
    return figure


def build_summary_text(summary_lines: list[str]) -> str:
    return "\n".join(summary_lines)


def save_figure(figure: plt.Figure, output_path: Path) -> None:
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _event_group_palette(count: int) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return ["#4c78a8"]
    colors: list[str] = []
    for index in range(count):
        hue = float(index) / max(count, 1)
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.90)
        colors.append(f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}")
    return colors


def build_novelty_threshold_chart(feature_table: pd.DataFrame) -> alt.Chart:
    required_columns = ["time_sec", "novelty", "threshold"]
    if feature_table.empty or any(column not in feature_table.columns for column in required_columns):
        return (
            alt.Chart(pd.DataFrame({"time_sec": [], "curve_value": []}))
            .mark_line()
            .encode(
                x=alt.X("time_sec:Q", title="时间（秒）"),
                y=alt.Y("curve_value:Q", title="检测强度"),
            )
            .properties(height=185, title="新颖度与检测阈值")
        )

    curve_source = feature_table[required_columns].apply(pd.to_numeric, errors="coerce")
    curve_source = curve_source.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    max_curve_points = 6_000
    if len(curve_source) > max_curve_points:
        quotient, remainder = divmod(len(curve_source), max_curve_points)
        group_sizes = np.full(max_curve_points, quotient, dtype=np.int64)
        group_sizes[:remainder] += 1
        group_starts = np.concatenate(
            [
                np.zeros(1, dtype=np.int64),
                np.cumsum(group_sizes[:-1], dtype=np.int64),
            ]
        )
        values = curve_source.to_numpy(dtype=np.float64)
        curve_source = pd.DataFrame(
            np.add.reduceat(values, group_starts, axis=0) / group_sizes[:, None],
            columns=required_columns,
        )
    curve_source["time_label"] = [format_seconds(value) for value in curve_source["time_sec"]]
    curve_long = curve_source.melt(
        id_vars=["time_sec", "time_label"],
        value_vars=["novelty", "threshold"],
        var_name="curve_type",
        value_name="curve_value",
    )
    curve_long["curve_label"] = curve_long["curve_type"].map(
        {"novelty": "新颖度", "threshold": "检测阈值"}
    )
    return (
        alt.Chart(curve_long)
        .mark_line(strokeWidth=1.35)
        .encode(
            x=alt.X("time_sec:Q", title=None),
            y=alt.Y("curve_value:Q", title="检测强度"),
            color=alt.Color(
                "curve_label:N",
                title="检测曲线",
                scale=alt.Scale(
                    domain=["新颖度", "检测阈值"],
                    range=["#187b72", "#dc6b48"],
                ),
            ),
            strokeDash=alt.StrokeDash(
                "curve_label:N",
                legend=None,
                scale=alt.Scale(
                    domain=["新颖度", "检测阈值"],
                    range=[[1, 0], [6, 4]],
                ),
            ),
            tooltip=[
                alt.Tooltip("time_label:N", title="时间"),
                alt.Tooltip("curve_label:N", title="曲线"),
                alt.Tooltip("curve_value:Q", title="值", format=".3f"),
            ],
        )
        .properties(height=185, title="新颖度与检测阈值")
    )


def build_interactive_novelty_chart(feature_table: pd.DataFrame, event_table: pd.DataFrame) -> alt.Chart:
    selector = alt.selection_point(
        name="event_pick",
        fields=["event_id"],
        on="click",
        empty=False,
    )

    event_source = event_table.copy()
    if event_source.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_point().encode(x="x:Q", y="y:Q")

    if "manual_time_sec" in event_source.columns:
        event_source["display_time_sec"] = pd.to_numeric(
            event_source["manual_time_sec"],
            errors="coerce",
        ).fillna(pd.to_numeric(event_source["time_sec"], errors="coerce"))
    else:
        event_source["display_time_sec"] = pd.to_numeric(
            event_source["time_sec"],
            errors="coerce",
        )
    if "manual_time_label" in event_source.columns:
        manual_labels = event_source["manual_time_label"].astype(str).str.strip()
        event_source["display_time_label"] = manual_labels.where(
            manual_labels != "",
            event_source["time_label"].astype(str),
        )
    else:
        event_source["display_time_label"] = event_source["time_label"].astype(str)

    event_source["boundary_type"] = np.where(
        event_source["is_major_boundary"],
        "候选边界",
        "普通事件",
    )
    if "interaction_priority_label" not in event_source.columns:
        event_source["interaction_priority_label"] = np.where(
            event_source["is_major_boundary"],
            "重点变化",
            "一般变化",
        )
    if "interaction_priority" not in event_source.columns:
        event_source["interaction_priority"] = np.where(
            event_source["is_major_boundary"],
            "strong",
            "medium",
        )
    if "similarity_group_label" not in event_source.columns:
        event_source["similarity_group_label"] = [f"相似组 {index}" for index in range(1, len(event_source) + 1)]
    if "similarity_group_size" not in event_source.columns:
        event_source["similarity_group_size"] = 1
    if "max_similarity_in_group" not in event_source.columns:
        event_source["max_similarity_in_group"] = 1.0

    size_map = {"strong": 210, "medium": 135, "weak": 78}
    opacity_map = {"strong": 0.98, "medium": 0.82, "weak": 0.42}
    event_source["point_size"] = event_source["interaction_priority"].map(size_map).fillna(135)
    event_source["point_opacity"] = event_source["interaction_priority"].map(opacity_map).fillna(0.82)
    similarity_domain = list(dict.fromkeys(event_source["similarity_group_label"].astype(str).tolist()))
    similarity_range = _event_group_palette(len(similarity_domain))

    point_chart = (
        alt.Chart(event_source)
        .mark_point(filled=True, size=110, stroke="#ffffff", strokeWidth=0.45)
        .encode(
            x=alt.X("display_time_sec:Q", title="时间（秒）"),
            y=alt.Y("strength:Q", title="事件强度"),
            size=alt.Size("point_size:Q", legend=None),
            opacity=alt.Opacity("point_opacity:Q", legend=None),
            shape=alt.Shape(
                "boundary_type:N",
                title="事件类型",
                scale=alt.Scale(
                    domain=["普通事件", "候选边界"],
                    range=["circle", "diamond"],
                ),
            ),
            color=alt.condition(
                selector,
                alt.value("#d62828"),
                alt.Color(
                    "similarity_group_label:N",
                    title="相似事件组",
                    scale=alt.Scale(
                        domain=similarity_domain,
                        range=similarity_range,
                    ),
                ),
            ),
            tooltip=[
                alt.Tooltip("event_id:Q", title="事件"),
                alt.Tooltip("display_time_label:N", title="当前时间"),
                alt.Tooltip("strength:Q", title="强度", format=".3f"),
                alt.Tooltip("prominence:Q", title="显著度", format=".3f"),
                alt.Tooltip("similarity_group_label:N", title="相似组"),
                alt.Tooltip("similarity_group_size:Q", title="组内事件数"),
                alt.Tooltip("max_similarity_in_group:Q", title="组内最高相似度", format=".0%"),
                alt.Tooltip("interaction_priority_label:N", title="交互层级"),
                alt.Tooltip("boundary_type:N", title="类型"),
                alt.Tooltip("auto_labels:N", title="候选标签"),
            ],
        )
        .add_params(selector)
    )

    return point_chart.properties(height=255, title="可点击事件点（颜色表示相似事件组）")


def _prepare_feature_curve_data(
    feature_table: pd.DataFrame,
    selected_features: list[str],
    feature_labels: dict[str, str],
    normalize: bool,
    max_time_points: int = 6_000,
) -> pd.DataFrame:
    valid_features = [name for name in selected_features if name in feature_table.columns]
    if feature_table.empty or not valid_features:
        return pd.DataFrame(columns=["time_sec", "time_label", "feature_name", "value", "feature_label"])

    times = pd.to_numeric(feature_table["time_sec"], errors="coerce").to_numpy(dtype=np.float64)
    values = feature_table[valid_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    times = np.nan_to_num(times, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize:
        means = np.mean(values, axis=0)
        standard_deviations = np.std(values, axis=0)
        standard_deviations = np.where(standard_deviations < 1e-8, 1.0, standard_deviations)
        values = (values - means[None, :]) / standard_deviations[None, :]

    safe_max_points = max(2, int(max_time_points))
    if times.size > safe_max_points:
        quotient, remainder = divmod(times.size, safe_max_points)
        group_sizes = np.full(safe_max_points, quotient, dtype=np.int64)
        group_sizes[:remainder] += 1
        group_starts = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(group_sizes[:-1], dtype=np.int64)]
        )
        times = np.add.reduceat(times, group_starts) / group_sizes
        values = np.add.reduceat(values, group_starts, axis=0) / group_sizes[:, None]

    feature_names = np.asarray(valid_features, dtype=object)
    long_frame = pd.DataFrame(
        {
            "time_sec": np.repeat(times, feature_names.size),
            "feature_name": np.tile(feature_names, times.size),
            "value": values.reshape(-1),
        }
    )
    long_frame["time_label"] = [format_seconds(value) for value in long_frame["time_sec"]]
    long_frame["feature_label"] = long_frame["feature_name"].map(feature_labels).fillna(long_frame["feature_name"])
    return long_frame


def build_feature_curve_chart(
    feature_table: pd.DataFrame,
    selected_features: list[str],
    feature_labels: dict[str, str],
    normalize: bool = True,
) -> alt.Chart:
    if not selected_features:
        selected_features = ["rms", "spectral_flux", "novelty"]

    long_df = _prepare_feature_curve_data(
        feature_table,
        selected_features,
        feature_labels,
        normalize=normalize,
    )

    chart = (
        alt.Chart(long_df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("time_sec:Q", title="时间（秒）"),
            y=alt.Y("value:Q", title="归一化特征值" if normalize else "特征值"),
            color=alt.Color("feature_label:N", title="特征"),
            tooltip=[
                alt.Tooltip("time_label:N", title="时间"),
                alt.Tooltip("feature_label:N", title="特征"),
                alt.Tooltip("value:Q", title="值", format=".4f"),
            ],
        )
        .properties(height=320, title="特征曲线")
    )
    return chart


def extract_selected_point_value(
    event_state: Any,
    selection_name: str,
    field_name: str,
) -> int | float | str | None:
    if not event_state:
        return None

    selection = None
    if isinstance(event_state, dict):
        selection = event_state.get("selection")
    else:
        selection = getattr(event_state, "selection", None)
    if selection is None:
        return None

    payload = selection.get(selection_name) if hasattr(selection, "get") else None
    if payload is None:
        payload = getattr(selection, selection_name, None)
    if payload is None:
        return None

    if isinstance(payload, dict):
        value = payload.get(field_name)
        if isinstance(value, list) and value:
            return value[0]
        if value is not None:
            return value
        if len(payload) == 1:
            only_value = next(iter(payload.values()))
            if isinstance(only_value, list) and only_value:
                return only_value[0]
            return only_value

    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict) and field_name in first:
            return first[field_name]
        return first

    return None
