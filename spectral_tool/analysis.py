from __future__ import annotations

import hashlib
import io
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Literal

import audioread
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from scipy.fft import rfft
from scipy.signal import find_peaks, get_window, resample_poly

ChannelMode = Literal["mix", "left", "right"]
CANDIDATE_BOUNDARY_LABEL = "候选段落边界"
ANALYSIS_METHOD_VERSION = "soundscape-method-2.2"

EVENT_LABEL_VOCAB = [
    "高频扩展",
    "噪声侵入",
    "稳态持续",
    "材料聚集",
    "消散",
    "新事件出现",
    CANDIDATE_BOUNDARY_LABEL,
    "音色突变",
    "能量增强",
    "能量减弱",
]

SECTION_LABEL_VOCAB = [
    "稳态持续",
    "材料聚集",
    "消散",
    "高频扩展",
    "噪声侵入",
    "能量增强",
    "能量减弱",
    "频谱扩散",
    "频谱集中",
    "纹理活跃",
]

FEATURE_COLUMN_LABELS = {
    "rms": "RMS",
    "spectral_centroid_hz": "Spectral Centroid",
    "low_band_ratio": "Low-band Energy Ratio",
    "mid_band_ratio": "Mid-band Energy Ratio",
    "band_energy_ratio": "Band Energy Ratio",
    "high_band_ratio": "High Frequency Ratio",
    "rolloff_hz": "Spectral Rolloff",
    "flatness": "Spectral Flatness",
    "spectral_flux": "Spectral Flux",
    "onset_strength": "Onset Strength",
    "spectral_entropy": "Spectral Entropy",
    "novelty_short_scale": "Short-scale Novelty",
    "novelty_medium_scale": "Medium-scale Novelty",
    "novelty_long_scale": "Long-scale Novelty",
    "multiscale_consensus": "Multi-scale Consensus",
    "novelty": "Novelty",
}


@dataclass(slots=True)
class AnalysisConfig:
    target_sr: int | None = None
    n_fft: int = 4096
    hop_length: int = 1024
    n_bands: int = 128
    smooth_sigma: float = 1.6
    threshold_sigma: float = 1.0
    prominence_sigma: float = 0.8
    min_event_distance_sec: float = 5.0
    context_window_sec: float = 3.0
    novelty_weight_cosine: float = 0.40
    novelty_weight_flux: float = 0.25
    novelty_weight_onset: float = 0.20
    novelty_weight_rms: float = 0.15
    event_model_preset: str = "balanced_default"
    multiscale_enabled: bool = True
    multiscale_weight: float = 0.35
    multiscale_factors: tuple[float, float, float] = (0.5, 1.0, 2.0)
    consensus_z_threshold: float = 0.65
    spectrogram_dynamic_range_db: float = 90.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def format_seconds(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    minutes = int(total_seconds // 60)
    secs = total_seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def join_labels(labels: list[str]) -> str:
    return " | ".join(_unique_labels(labels))


def split_label_text(label_text: str) -> list[str]:
    if not label_text:
        return []
    parts = [part.strip() for part in label_text.replace("；", "|").replace("、", "|").split("|")]
    return [part for part in parts if part]


def _unique_labels(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for label in labels:
        if label and label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - mean) / std


def _safe_rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))


def _magnitude_to_relative_db(
    magnitude: np.ndarray,
    dynamic_range_db: float,
) -> np.ndarray:
    """Convert a magnitude array to peak-relative dB without brightening silence."""
    values = np.nan_to_num(
        np.asarray(magnitude, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    values = np.maximum(values, 0.0)
    dynamic_range = max(20.0, float(dynamic_range_db))
    peak = float(np.max(values)) if values.size else 0.0
    if peak <= np.finfo(np.float32).tiny:
        return np.full(values.shape, -dynamic_range, dtype=np.float32)

    minimum = peak * (10.0 ** (-dynamic_range / 20.0))
    relative_db = 20.0 * np.log10(np.maximum(values, minimum) / peak)
    return np.clip(relative_db, -dynamic_range, 0.0).astype(np.float32)


def _resolve_novelty_weights(config: AnalysisConfig) -> tuple[float, float, float, float]:
    weights = np.array(
        [
            float(config.novelty_weight_cosine),
            float(config.novelty_weight_flux),
            float(config.novelty_weight_onset),
            float(config.novelty_weight_rms),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)):
        return (0.40, 0.25, 0.20, 0.15)

    weights = np.clip(weights, a_min=0.0, a_max=None)
    total = float(np.sum(weights))
    if total <= 1e-8:
        return (0.40, 0.25, 0.20, 0.15)

    normalized = weights / total
    return tuple(float(value) for value in normalized)


def _resolve_source_name(source: str | Path | BinaryIO) -> str:
    if isinstance(source, Path):
        return source.name
    if isinstance(source, str):
        return Path(source).name
    name = getattr(source, "name", None)
    if isinstance(name, str) and name:
        return Path(name).name
    return "audio_input"


def _write_temp_audio_file(source: BinaryIO) -> tuple[str, str | None]:
    source.seek(0)
    data = source.read()
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("音频输入必须是字节流或文件路径。")

    suffix = Path(_resolve_source_name(source)).suffix or ".bin"
    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temporary.write(data)
        temporary.flush()
    finally:
        temporary.close()
    return temporary.name, suffix


def _read_with_audioread(source: str | Path | BinaryIO) -> tuple[np.ndarray, int]:
    temp_path: str | None = None
    audio_path: str
    if isinstance(source, (str, Path)):
        audio_path = str(source)
    else:
        audio_path, _ = _write_temp_audio_file(source)
        temp_path = audio_path

    try:
        with audioread.audio_open(audio_path) as handle:
            chunks: list[np.ndarray] = []
            channel_count = int(handle.channels)
            sample_rate = int(handle.samplerate)

            for chunk in handle:
                pcm = np.frombuffer(chunk, dtype="<i2")
                if pcm.size == 0:
                    continue
                trimmed = pcm[: pcm.size - (pcm.size % channel_count)]
                if trimmed.size == 0:
                    continue
                frames = trimmed.reshape(-1, channel_count).astype(np.float32) / 32768.0
                chunks.append(frames)

        if not chunks:
            raise RuntimeError("无法从音频中读取有效的 PCM 数据。")

        audio = np.concatenate(chunks, axis=0)
        return audio.T.astype(np.float32), sample_rate
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _read_audio(source: str | Path | BinaryIO) -> tuple[np.ndarray, int]:
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        audio, sr = sf.read(source, always_2d=True, dtype="float32")
        return audio.T.astype(np.float32), int(sr)
    except Exception:
        try:
            return _read_with_audioread(source)
        except Exception as audioread_error:
            raise RuntimeError(
                "无法读取该音频文件。请尝试 WAV、FLAC、AIFF、MP3 或 M4A。"
            ) from audioread_error


def _load_audio(source: str | Path | BinaryIO, target_sr: int | None) -> tuple[np.ndarray, int]:
    channels, sr = _read_audio(source)

    if target_sr and target_sr != sr:
        gcd = math.gcd(int(sr), int(target_sr))
        up = int(target_sr) // gcd
        down = int(sr) // gcd
        resampled_channels: list[np.ndarray] = []
        for channel in channels:
            resampled = resample_poly(channel, up=up, down=down).astype(np.float32)
            resampled_channels.append(resampled)
        min_length = min(channel.shape[0] for channel in resampled_channels)
        channels = np.vstack([channel[:min_length] for channel in resampled_channels])
        sr = int(target_sr)

    return channels.astype(np.float32), int(sr)


def _build_channel_rms_envelope(audio: np.ndarray, sr: int, block_size: int = 65_536) -> dict[str, object]:
    rms_blocks: list[np.ndarray] = []
    time_blocks: list[float] = []
    for start in range(0, audio.shape[1], block_size):
        stop = min(audio.shape[1], start + block_size)
        block = audio[:, start:stop]
        rms_blocks.append(np.sqrt(np.mean(np.square(block), axis=1, dtype=np.float64)).astype(np.float32))
        time_blocks.append((start + (stop - start) / 2.0) / sr)
    rms = np.stack(rms_blocks, axis=1) if rms_blocks else np.empty((audio.shape[0], 0), dtype=np.float32)
    return {
        "times": np.asarray(time_blocks, dtype=np.float32),
        "rms": rms,
        "channel_count": int(audio.shape[0]),
    }


def _read_selected_soundfile(
    source: str | Path | BinaryIO,
    channel_mode: ChannelMode,
    target_sr: int | None,
) -> tuple[np.ndarray, int, dict[str, object]]:
    position = source.tell() if not isinstance(source, (str, Path)) and hasattr(source, "tell") else None
    if position is not None:
        source.seek(0)

    try:
        with sf.SoundFile(source, mode="r") as handle:
            source_sr = int(handle.samplerate)
            channel_count = int(handle.channels)
            frame_count = int(len(handle))
            if source_sr <= 0 or channel_count <= 0 or frame_count <= 0:
                raise ValueError("音频为空或缺少有效的采样信息。")

            rms_blocks: list[np.ndarray] = []
            time_blocks: list[float] = []

            def select_channel(block: np.ndarray) -> np.ndarray:
                if channel_count == 1:
                    return block[:, 0]
                if channel_mode == "left":
                    return block[:, 0]
                if channel_mode == "right":
                    return block[:, 1]
                return np.mean(block, axis=1, dtype=np.float32)

            def append_envelope(core_block: np.ndarray, absolute_start: int) -> None:
                envelope_block_size = 65_536
                for local_start in range(0, core_block.shape[0], envelope_block_size):
                    local_stop = min(core_block.shape[0], local_start + envelope_block_size)
                    envelope_block = core_block[local_start:local_stop]
                    rms_blocks.append(
                        np.sqrt(
                            np.mean(np.square(envelope_block), axis=0, dtype=np.float64)
                        ).astype(np.float32)
                    )
                    time_blocks.append(
                        (absolute_start + local_start + (local_stop - local_start) / 2.0) / source_sr
                    )

            analysis_sr = int(target_sr) if target_sr and target_sr != source_sr else source_sr
            if analysis_sr == source_sr:
                selected = np.empty(frame_count, dtype=np.float32)
                cursor = 0
                block_size = 65_536
                while cursor < frame_count:
                    block = handle.read(
                        frames=min(block_size, frame_count - cursor),
                        dtype="float32",
                        always_2d=True,
                    )
                    if block.size == 0:
                        break
                    stop = cursor + block.shape[0]
                    selected[cursor:stop] = select_channel(block)
                    append_envelope(block, cursor)
                    cursor = stop
                selected = selected[:cursor]
            else:
                gcd = math.gcd(source_sr, analysis_sr)
                up = analysis_sr // gcd
                down = source_sr // gcd
                ratio = analysis_sr / source_sr
                target_frame_count = (frame_count * up + down - 1) // down
                selected = np.empty(target_frame_count, dtype=np.float32)
                core_size = 1_048_576
                overlap = 4_096
                for core_start in range(0, frame_count, core_size):
                    core_end = min(frame_count, core_start + core_size)
                    extended_start = max(0, core_start - overlap)
                    # Keep every block on the same polyphase grid as a whole-file resample.
                    extended_start -= extended_start % down
                    extended_end = min(frame_count, core_end + overlap)
                    handle.seek(extended_start)
                    extended_block = handle.read(
                        frames=extended_end - extended_start,
                        dtype="float32",
                        always_2d=True,
                    )
                    core_local_start = core_start - extended_start
                    core_local_end = core_local_start + (core_end - core_start)
                    append_envelope(extended_block[core_local_start:core_local_end], core_start)

                    resampled = resample_poly(
                        select_channel(extended_block),
                        up=up,
                        down=down,
                    ).astype(np.float32)
                    output_start = int(round(core_start * ratio))
                    output_end = (
                        target_frame_count
                        if core_end == frame_count
                        else int(round(core_end * ratio))
                    )
                    extended_output_start = (extended_start * up) // down
                    local_output_start = output_start - extended_output_start
                    local_output_end = local_output_start + (output_end - output_start)
                    core_resampled = resampled[local_output_start:local_output_end]
                    expected_size = output_end - output_start
                    if core_resampled.size < expected_size:
                        core_resampled = np.pad(core_resampled, (0, expected_size - core_resampled.size), mode="edge")
                    selected[output_start:output_end] = core_resampled[:expected_size]

            rms = (
                np.stack(rms_blocks, axis=1)
                if rms_blocks
                else np.empty((channel_count, 0), dtype=np.float32)
            )
            envelope = {
                "times": np.asarray(time_blocks, dtype=np.float32),
                "rms": rms,
                "channel_count": channel_count,
            }
            return selected, analysis_sr, envelope
    finally:
        if position is not None:
            source.seek(position)


def _load_analysis_audio(
    source: str | Path | BinaryIO,
    target_sr: int | None,
    channel_mode: ChannelMode,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, object]]:
    try:
        selected, sr, channel_envelope = _read_selected_soundfile(source, channel_mode, target_sr)
        analysis_audio = selected.reshape(1, -1)
        return analysis_audio, selected, sr, channel_envelope
    except sf.LibsndfileError:
        audio, sr = _load_audio(source, target_sr)
        selected = _select_channel(audio, channel_mode).astype(np.float32)
        channel_envelope = _build_channel_rms_envelope(audio, sr)
        return audio, selected, sr, channel_envelope


def _select_channel(audio: np.ndarray, mode: ChannelMode) -> np.ndarray:
    if audio.shape[0] == 1:
        return audio[0]
    if mode == "left":
        return audio[0]
    if mode == "right":
        return audio[1]
    return np.mean(audio, axis=0)


def build_audio_excerpt_wav(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    channel_mode: ChannelMode = "mix",
) -> bytes:
    start_sample = max(0, int(start_sec * sr))
    end_sample = max(start_sample + 1, int(end_sec * sr))

    if audio.ndim == 1:
        clip = audio[start_sample:end_sample]
        clip_to_write = clip.reshape(-1, 1)
    else:
        if channel_mode == "left" and audio.shape[0] > 1:
            clip = audio[0, start_sample:end_sample]
            clip_to_write = clip.reshape(-1, 1)
        elif channel_mode == "right" and audio.shape[0] > 1:
            clip = audio[1, start_sample:end_sample]
            clip_to_write = clip.reshape(-1, 1)
        elif channel_mode == "mix":
            clip = np.mean(audio[:, start_sample:end_sample], axis=0)
            clip_to_write = clip.reshape(-1, 1)
        else:
            clip_to_write = audio[:, start_sample:end_sample].T

    buffer = io.BytesIO()
    sf.write(buffer, clip_to_write, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _spectrogram_params(sample_count: int, config: AnalysisConfig) -> tuple[int, int]:
    if sample_count < 2:
        raise ValueError("音频过短，至少需要 2 个采样点才能进行频谱分析。")
    n_fft = min(max(2, int(config.n_fft)), sample_count)
    hop = min(max(1, int(config.hop_length)), max(1, n_fft // 4))
    if hop >= n_fft:
        hop = max(1, n_fft // 2)
    return int(n_fft), int(hop)


def _iter_magnitude_blocks(
    samples: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    block_frame_count: int = 384,
):
    """Yield consecutive STFT blocks while preserving one global frame grid."""
    if samples.size < 2:
        raise ValueError("音频过短，至少需要 2 个采样点才能进行频谱分析。")

    safe_n_fft = min(max(2, int(n_fft)), samples.size)
    safe_hop = min(max(1, int(hop)), max(1, safe_n_fft // 2))
    frame_count = 1 + max(0, (samples.size - safe_n_fft) // safe_hop)
    window = get_window("hann", safe_n_fft, fftbins=True).astype(np.float32)
    spectrum_scale = 1.0 / max(float(np.sum(window)), 1e-12)
    freqs = np.fft.rfftfreq(safe_n_fft, d=1.0 / sr).astype(np.float32)

    for first_frame in range(0, frame_count, block_frame_count):
        frames_in_block = min(block_frame_count, frame_count - first_frame)
        first_sample = first_frame * safe_hop
        required_samples = safe_n_fft + (frames_in_block - 1) * safe_hop
        sample_block = np.asarray(
            samples[first_sample : first_sample + required_samples],
            dtype=np.float32,
        )
        frames = np.lib.stride_tricks.as_strided(
            sample_block,
            shape=(frames_in_block, safe_n_fft),
            strides=(sample_block.strides[0] * safe_hop, sample_block.strides[0]),
            writeable=False,
        )
        windowed = frames * window[None, :]
        magnitude = (np.abs(rfft(windowed, axis=1)) * spectrum_scale).T.astype(np.float32)
        frame_indices = first_frame + np.arange(frames_in_block, dtype=np.float64)
        times = ((frame_indices * safe_hop + safe_n_fft / 2.0) / sr).astype(np.float32)
        yield freqs, times, magnitude


def _aggregate_frequency_bins(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    group_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    row_count = int(spectrum.shape[0])
    safe_group_count = max(1, int(group_count))
    if row_count <= safe_group_count:
        return freqs.astype(np.float32), spectrum.astype(np.float32)

    quotient, remainder = divmod(row_count, safe_group_count)
    group_sizes = np.full(safe_group_count, quotient, dtype=np.int64)
    group_sizes[:remainder] += 1
    group_starts = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(group_sizes[:-1], dtype=np.int64)]
    )

    frequency_sums = np.add.reduceat(np.asarray(freqs, dtype=np.float64), group_starts)
    spectrum_sums = np.add.reduceat(
        np.asarray(spectrum, dtype=np.float32),
        group_starts,
        axis=0,
        dtype=np.float64,
    )
    aggregated_freqs = (frequency_sums / group_sizes).astype(np.float32)
    aggregated_spectrum = (spectrum_sums / group_sizes[:, None]).astype(np.float32)
    return aggregated_freqs, aggregated_spectrum


def _aggregate_frequency_peaks(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    group_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool display frequencies by peak magnitude so narrow partials stay visible."""
    row_count = int(spectrum.shape[0])
    safe_group_count = max(1, int(group_count))
    if row_count <= safe_group_count:
        return freqs.astype(np.float32), spectrum.astype(np.float32)

    quotient, remainder = divmod(row_count, safe_group_count)
    group_sizes = np.full(safe_group_count, quotient, dtype=np.int64)
    group_sizes[:remainder] += 1
    group_starts = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(group_sizes[:-1], dtype=np.int64)]
    )
    frequency_sums = np.add.reduceat(np.asarray(freqs, dtype=np.float64), group_starts)
    aggregated_freqs = (frequency_sums / group_sizes).astype(np.float32)
    aggregated_spectrum = np.maximum.reduceat(
        np.asarray(spectrum, dtype=np.float32),
        group_starts,
        axis=0,
    )
    return aggregated_freqs, aggregated_spectrum.astype(np.float32)


def _framewise_cosine_distance(log_spectrum: np.ndarray) -> np.ndarray:
    frame_count = log_spectrum.shape[1]
    distances = np.zeros(frame_count, dtype=np.float32)
    if frame_count < 2:
        return distances

    previous = log_spectrum[:, :-1]
    current = log_spectrum[:, 1:]
    previous_norm = np.linalg.norm(previous, axis=0)
    current_norm = np.linalg.norm(current, axis=0)
    denominator = np.maximum(previous_norm * current_norm, 1e-8)
    similarity = np.sum(previous * current, axis=0) / denominator
    distances[1:] = 1.0 - np.clip(similarity, -1.0, 1.0)
    return distances


def _framewise_rolloff(freqs: np.ndarray, magnitude: np.ndarray, fraction: float = 0.85) -> np.ndarray:
    power = np.square(np.maximum(np.asarray(magnitude, dtype=np.float32), 0.0))
    cumulative = np.cumsum(power, axis=0, dtype=np.float64)
    totals = cumulative[-1, :]
    thresholds = totals * float(np.clip(fraction, 0.0, 1.0))
    mask = cumulative >= thresholds[None, :]
    indices = np.argmax(mask, axis=0)
    rolloff = freqs[indices].astype(np.float32)
    rolloff[totals <= np.finfo(np.float32).tiny] = 0.0
    return rolloff


def _framewise_flatness(magnitude: np.ndarray) -> np.ndarray:
    power = np.square(np.maximum(np.asarray(magnitude, dtype=np.float32), 0.0))
    totals = np.sum(power, axis=0, dtype=np.float64)
    safe_power = np.maximum(power, np.finfo(np.float32).tiny)
    geometric_mean = np.exp(np.mean(np.log(safe_power), axis=0, dtype=np.float64))
    arithmetic_mean = np.mean(power, axis=0, dtype=np.float64)
    flatness = np.divide(
        geometric_mean,
        arithmetic_mean,
        out=np.zeros_like(arithmetic_mean),
        where=arithmetic_mean > np.finfo(np.float32).tiny,
    )
    flatness[totals <= np.finfo(np.float32).tiny] = 0.0
    return np.clip(flatness, 0.0, 1.0).astype(np.float32)


def _framewise_spectral_entropy(magnitude: np.ndarray) -> np.ndarray:
    power = np.square(np.maximum(np.asarray(magnitude, dtype=np.float32), 0.0))
    totals = np.sum(power, axis=0, keepdims=True, dtype=np.float64)
    distribution = np.divide(
        power,
        totals,
        out=np.zeros_like(power, dtype=np.float64),
        where=totals > np.finfo(np.float32).tiny,
    )
    denominator = max(math.log(max(2, power.shape[0])), 1e-8)
    entropy = -np.sum(
        distribution * np.log(np.maximum(distribution, np.finfo(np.float64).tiny)),
        axis=0,
    ) / denominator
    return np.clip(entropy, 0.0, 1.0).astype(np.float32)


@lru_cache(maxsize=32)
def _stft_rms_energy_scale(n_fft: int) -> float:
    window = get_window("hann", n_fft, fftbins=True).astype(np.float64)
    coherent_sum = float(np.sum(window))
    window_energy = float(np.sum(np.square(window)))
    return (coherent_sum * coherent_sum) / max(float(n_fft) * window_energy, 1e-12)


def _framewise_rms_from_magnitude(magnitude: np.ndarray, n_fft: int) -> np.ndarray:
    """Recover window-normalized frame RMS from the scaled one-sided STFT."""
    squared = np.square(np.asarray(magnitude, dtype=np.float64))
    if squared.shape[0] == 1:
        one_sided_energy = squared[0]
    elif n_fft % 2 == 0:
        one_sided_energy = squared[0] + squared[-1] + 2.0 * np.sum(squared[1:-1], axis=0)
    else:
        one_sided_energy = squared[0] + 2.0 * np.sum(squared[1:], axis=0)
    rms_squared = one_sided_energy * _stft_rms_energy_scale(int(n_fft))
    return np.sqrt(np.maximum(rms_squared, 0.0)).astype(np.float32)


def _single_scale_change_curve(
    samples: np.ndarray,
    sr: int,
    target_times: np.ndarray,
    n_fft: int,
    hop: int,
    config: AnalysisConfig,
) -> np.ndarray:
    times_parts: list[np.ndarray] = []
    cosine_parts: list[np.ndarray] = []
    flux_parts: list[np.ndarray] = []
    previous_log_band: np.ndarray | None = None
    for freqs, block_times, magnitude in _iter_magnitude_blocks(samples, sr, n_fft, hop):
        _, band_spectrum = _aggregate_frequency_bins(freqs, magnitude, group_count=config.n_bands)
        log_bands = np.log1p(np.maximum(band_spectrum, 1e-10)).astype(np.float32)
        cosine_distance = _framewise_cosine_distance(log_bands)
        spectral_flux = np.zeros(log_bands.shape[1], dtype=np.float32)
        if previous_log_band is not None and log_bands.shape[1]:
            previous_norm = max(float(np.linalg.norm(previous_log_band)), 1e-8)
            current_norm = max(float(np.linalg.norm(log_bands[:, 0])), 1e-8)
            similarity = float(np.dot(previous_log_band, log_bands[:, 0]) / (previous_norm * current_norm))
            cosine_distance[0] = 1.0 - float(np.clip(similarity, -1.0, 1.0))
            spectral_flux[0] = float(np.mean(np.maximum(log_bands[:, 0] - previous_log_band, 0.0)))
        if log_bands.shape[1] > 1:
            spectral_flux[1:] = np.mean(np.maximum(np.diff(log_bands, axis=1), 0.0), axis=0)
        if log_bands.shape[1]:
            previous_log_band = log_bands[:, -1].copy()
        times_parts.append(block_times)
        cosine_parts.append(cosine_distance)
        flux_parts.append(spectral_flux)

    times = np.concatenate(times_parts).astype(np.float32)
    cosine_distance = np.concatenate(cosine_parts).astype(np.float32)
    spectral_flux = np.concatenate(flux_parts).astype(np.float32)

    change_curve = 0.68 * _zscore(cosine_distance) + 0.32 * _zscore(spectral_flux)
    change_curve = gaussian_filter1d(change_curve.astype(np.float32), sigma=max(config.smooth_sigma, 0.1))
    if times.size == 0 or target_times.size == 0:
        return np.zeros(target_times.shape, dtype=np.float32)
    return np.interp(target_times, times, change_curve, left=change_curve[0], right=change_curve[-1]).astype(np.float32)


def _build_multiscale_change_bundle(
    samples: np.ndarray,
    sr: int,
    target_times: np.ndarray,
    base_n_fft: int,
    base_hop: int,
    primary_novelty: np.ndarray,
    config: AnalysisConfig,
    base_change_curve: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if not config.multiscale_enabled or target_times.size == 0:
        normalized = _zscore(primary_novelty)
        return {
            "short": normalized,
            "medium": normalized,
            "long": normalized,
            "combined": normalized,
            "consensus": np.ones(target_times.shape, dtype=np.float32),
        }

    factors = sorted(float(value) for value in config.multiscale_factors)
    curves: list[np.ndarray] = []
    curve_cache: dict[tuple[int, int], np.ndarray] = {}
    for factor in factors:
        scale_n_fft = int(np.clip(round(base_n_fft * factor), 512, 16384))
        scale_hop = int(np.clip(round(base_hop * factor), 128, max(128, scale_n_fft // 2)))
        scale_key = (scale_n_fft, scale_hop)
        if scale_key in curve_cache:
            curve = curve_cache[scale_key]
        elif scale_key == (base_n_fft, base_hop) and base_change_curve is not None:
            curve = base_change_curve
            curve_cache[scale_key] = curve
        else:
            curve = _single_scale_change_curve(
                samples=samples,
                sr=sr,
                target_times=target_times,
                n_fft=scale_n_fft,
                hop=scale_hop,
                config=config,
            )
            curve_cache[scale_key] = curve
        curves.append(_zscore(curve))

    while len(curves) < 3:
        curves.append(curves[-1] if curves else _zscore(primary_novelty))
    scale_matrix = np.vstack(curves[:3]).astype(np.float32)
    combined = np.mean(scale_matrix, axis=0).astype(np.float32)
    consensus = np.mean(scale_matrix >= float(config.consensus_z_threshold), axis=0).astype(np.float32)
    return {
        "short": scale_matrix[0],
        "medium": scale_matrix[1],
        "long": scale_matrix[2],
        "combined": combined,
        "consensus": consensus,
    }


def _framewise_centroid_and_bandwidth(freqs: np.ndarray, magnitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.maximum(np.asarray(magnitude, dtype=np.float32), 0.0)
    totals = np.sum(weights, axis=0, dtype=np.float64)
    weighted_frequency = np.sum(freqs[:, None] * weights, axis=0, dtype=np.float64)
    centroid = np.divide(
        weighted_frequency,
        totals,
        out=np.zeros_like(totals),
        where=totals > np.finfo(np.float32).tiny,
    )
    weighted_variance = np.sum(
        np.square(freqs[:, None] - centroid[None, :]) * weights,
        axis=0,
        dtype=np.float64,
    )
    variance = np.divide(
        weighted_variance,
        totals,
        out=np.zeros_like(totals),
        where=totals > np.finfo(np.float32).tiny,
    )
    return centroid.astype(np.float32), np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)


def _framewise_band_energy_ratios(freqs: np.ndarray, magnitude: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    power = np.square(np.maximum(np.asarray(magnitude, dtype=np.float32), 0.0))
    total_energy = np.sum(power, axis=0, dtype=np.float64)
    low_mask = freqs < 250.0
    mid_mask = (freqs >= 250.0) & (freqs < 2000.0)
    high_mask = freqs >= 2000.0

    def band_ratio(mask: np.ndarray) -> np.ndarray:
        band_energy = np.sum(power[mask, :], axis=0, dtype=np.float64)
        return np.divide(
            band_energy,
            total_energy,
            out=np.zeros_like(total_energy),
            where=total_energy > np.finfo(np.float32).tiny,
        ).astype(np.float32)

    low_ratio = band_ratio(low_mask)
    mid_ratio = band_ratio(mid_mask)
    high_ratio = band_ratio(high_mask)
    band_energy_ratio = (high_ratio / np.maximum(low_ratio + mid_ratio, 1e-8)).astype(np.float32)
    return low_ratio, mid_ratio, high_ratio, band_energy_ratio


def _build_feature_table(
    times: np.ndarray,
    rms: np.ndarray,
    centroid: np.ndarray,
    low_band_ratio: np.ndarray,
    mid_band_ratio: np.ndarray,
    band_energy_ratio: np.ndarray,
    high_band_ratio: np.ndarray,
    bandwidth: np.ndarray,
    rolloff: np.ndarray,
    flatness: np.ndarray,
    spectral_flux: np.ndarray,
    onset_strength: np.ndarray,
    spectral_entropy: np.ndarray,
    novelty_short_scale: np.ndarray,
    novelty_medium_scale: np.ndarray,
    novelty_long_scale: np.ndarray,
    multiscale_consensus: np.ndarray,
    novelty: np.ndarray,
    threshold: np.ndarray,
) -> pd.DataFrame:
    frame_table = pd.DataFrame(
        {
            "time_sec": np.round(times.astype(np.float64), 6),
            "time_label": [format_seconds(value) for value in times],
            "rms": rms.astype(np.float32),
            "spectral_centroid_hz": centroid.astype(np.float32),
            "low_band_ratio": low_band_ratio.astype(np.float32),
            "mid_band_ratio": mid_band_ratio.astype(np.float32),
            "band_energy_ratio": band_energy_ratio.astype(np.float32),
            "high_band_ratio": high_band_ratio.astype(np.float32),
            "bandwidth_hz": bandwidth.astype(np.float32),
            "rolloff_hz": rolloff.astype(np.float32),
            "flatness": flatness.astype(np.float32),
            "spectral_flux": spectral_flux.astype(np.float32),
            "onset_strength": onset_strength.astype(np.float32),
            "spectral_entropy": spectral_entropy.astype(np.float32),
            "novelty_short_scale": novelty_short_scale.astype(np.float32),
            "novelty_medium_scale": novelty_medium_scale.astype(np.float32),
            "novelty_long_scale": novelty_long_scale.astype(np.float32),
            "multiscale_consensus": multiscale_consensus.astype(np.float32),
            "novelty": novelty.astype(np.float32),
            "threshold": threshold.astype(np.float32),
        }
    )
    return frame_table


def _build_novelty_curve(samples: np.ndarray, sr: int, config: AnalysisConfig) -> dict[str, np.ndarray | int]:
    n_fft, hop = _spectrogram_params(samples.size, config)
    total_frame_count = 1 + max(0, (samples.size - n_fft) // hop)
    display_time_bin_count = min(3600, total_frame_count)
    times_parts: list[np.ndarray] = []
    log_band_parts: list[np.ndarray] = []
    base_cosine_parts: list[np.ndarray] = []
    rms_parts: list[np.ndarray] = []
    centroid_parts: list[np.ndarray] = []
    bandwidth_parts: list[np.ndarray] = []
    rolloff_parts: list[np.ndarray] = []
    flatness_parts: list[np.ndarray] = []
    entropy_parts: list[np.ndarray] = []
    low_ratio_parts: list[np.ndarray] = []
    mid_ratio_parts: list[np.ndarray] = []
    high_ratio_parts: list[np.ndarray] = []
    band_ratio_parts: list[np.ndarray] = []
    spectral_flux_parts: list[np.ndarray] = []
    onset_parts: list[np.ndarray] = []
    previous_log_spectrum: np.ndarray | None = None
    previous_log_band: np.ndarray | None = None
    display_freqs = np.array([], dtype=np.float32)
    display_spectrum_sum: np.ndarray | None = None
    display_time_sum = np.zeros(display_time_bin_count, dtype=np.float64)
    display_bin_counts = np.zeros(display_time_bin_count, dtype=np.int64)
    processed_frame_count = 0

    for freqs, block_times, magnitude in _iter_magnitude_blocks(samples, sr, n_fft, hop):
        times_parts.append(block_times)
        _, band_spectrum = _aggregate_frequency_bins(freqs, magnitude, group_count=config.n_bands)
        log_bands = np.log1p(np.maximum(band_spectrum, 1e-10)).astype(np.float32)
        if config.event_model_preset == "timbre_soundscape":
            log_band_parts.append(log_bands)
        block_cosine = _framewise_cosine_distance(log_bands)
        block_flux = np.zeros(log_bands.shape[1], dtype=np.float32)
        if previous_log_band is not None and log_bands.shape[1]:
            previous_norm = max(float(np.linalg.norm(previous_log_band)), 1e-8)
            current_norm = max(float(np.linalg.norm(log_bands[:, 0])), 1e-8)
            similarity = float(
                np.dot(previous_log_band, log_bands[:, 0])
                / (previous_norm * current_norm)
            )
            block_cosine[0] = 1.0 - float(np.clip(similarity, -1.0, 1.0))
            block_flux[0] = float(np.mean(np.maximum(log_bands[:, 0] - previous_log_band, 0.0)))
        if log_bands.shape[1] > 1:
            block_flux[1:] = np.mean(np.maximum(np.diff(log_bands, axis=1), 0.0), axis=0)
        if log_bands.shape[1]:
            previous_log_band = log_bands[:, -1].copy()
        base_cosine_parts.append(block_cosine)
        spectral_flux_parts.append(block_flux)

        display_group_count = min(512, magnitude.shape[0])
        display_freqs, display_spectrum = _aggregate_frequency_peaks(
            freqs,
            magnitude,
            group_count=display_group_count,
        )
        if display_spectrum_sum is None:
            display_spectrum_sum = np.zeros(
                (display_spectrum.shape[0], display_time_bin_count),
                dtype=np.float64,
            )
        global_indices = processed_frame_count + np.arange(block_times.size, dtype=np.int64)
        display_bins = np.minimum(
            display_time_bin_count - 1,
            (global_indices * display_time_bin_count) // max(total_frame_count, 1),
        )
        unique_bins, group_starts, group_counts = np.unique(
            display_bins,
            return_index=True,
            return_counts=True,
        )
        display_spectrum_sum[:, unique_bins] += np.add.reduceat(
            np.square(display_spectrum),
            group_starts,
            axis=1,
            dtype=np.float64,
        )
        display_time_sum[unique_bins] += np.add.reduceat(
            block_times.astype(np.float64),
            group_starts,
        )
        display_bin_counts[unique_bins] += group_counts
        processed_frame_count += block_times.size

        centroid, bandwidth = _framewise_centroid_and_bandwidth(freqs, magnitude)
        low_band_ratio, mid_band_ratio, high_band_ratio, band_energy_ratio = (
            _framewise_band_energy_ratios(freqs, magnitude)
        )
        centroid_parts.append(centroid)
        bandwidth_parts.append(bandwidth)
        rolloff_parts.append(_framewise_rolloff(freqs, magnitude))
        flatness_parts.append(_framewise_flatness(magnitude))
        entropy_parts.append(_framewise_spectral_entropy(magnitude))
        low_ratio_parts.append(low_band_ratio)
        mid_ratio_parts.append(mid_band_ratio)
        high_ratio_parts.append(high_band_ratio)
        band_ratio_parts.append(band_energy_ratio)
        rms_parts.append(_framewise_rms_from_magnitude(magnitude, n_fft))

        log_magnitude = np.log1p(np.maximum(magnitude, 1e-10)).astype(np.float32)
        onset = np.zeros(log_magnitude.shape[1], dtype=np.float32)
        freq_weights = np.linspace(0.7, 1.6, log_magnitude.shape[0], dtype=np.float32)
        weight_total = float(np.sum(freq_weights))
        if previous_log_spectrum is not None and log_magnitude.shape[1]:
            first_diff = np.maximum(log_magnitude[:, 0] - previous_log_spectrum, 0.0)
            onset[0] = float(np.sum(first_diff * freq_weights) / weight_total)
        if log_magnitude.shape[1] > 1:
            positive_diff = np.maximum(np.diff(log_magnitude, axis=1), 0.0)
            onset[1:] = (
                np.sum(positive_diff * freq_weights[:, None], axis=0) / weight_total
            ).astype(np.float32)
        if log_magnitude.shape[1]:
            previous_log_spectrum = log_magnitude[:, -1].copy()
        onset_parts.append(onset)

    times = np.concatenate(times_parts).astype(np.float32)
    base_band_cosine_distance = np.concatenate(base_cosine_parts).astype(np.float32)
    rms = np.concatenate(rms_parts).astype(np.float32)
    centroid = np.concatenate(centroid_parts).astype(np.float32)
    bandwidth = np.concatenate(bandwidth_parts).astype(np.float32)
    rolloff = np.concatenate(rolloff_parts).astype(np.float32)
    flatness = np.concatenate(flatness_parts).astype(np.float32)
    spectral_entropy = np.concatenate(entropy_parts).astype(np.float32)
    low_band_ratio = np.concatenate(low_ratio_parts).astype(np.float32)
    mid_band_ratio = np.concatenate(mid_ratio_parts).astype(np.float32)
    high_band_ratio = np.concatenate(high_ratio_parts).astype(np.float32)
    band_energy_ratio = np.concatenate(band_ratio_parts).astype(np.float32)
    onset_strength = gaussian_filter1d(np.concatenate(onset_parts).astype(np.float32), sigma=1.0)
    spectral_flux = np.concatenate(spectral_flux_parts).astype(np.float32)

    if config.event_model_preset == "timbre_soundscape":
        log_bands = np.concatenate(log_band_parts, axis=1).astype(np.float32)
        descriptor_extras = np.vstack(
            [
                _zscore(np.log1p(np.maximum(centroid, 0.0))),
                _zscore(band_energy_ratio),
                _zscore(high_band_ratio),
                _zscore(flatness),
            ]
        ).astype(np.float32)
        cosine_distance = np.zeros(times.shape, dtype=np.float32)
        previous_descriptor: np.ndarray | None = None
        descriptor_block_size = 4096
        for start in range(0, times.size, descriptor_block_size):
            stop = min(times.size, start + descriptor_block_size)
            descriptor = np.vstack(
                [log_bands[:, start:stop], descriptor_extras[:, start:stop]]
            ).astype(np.float32)
            block_distance = _framewise_cosine_distance(descriptor)
            if previous_descriptor is not None and descriptor.shape[1]:
                previous_norm = max(float(np.linalg.norm(previous_descriptor)), 1e-8)
                current_norm = max(float(np.linalg.norm(descriptor[:, 0])), 1e-8)
                similarity = float(
                    np.dot(previous_descriptor, descriptor[:, 0]) / (previous_norm * current_norm)
                )
                block_distance[0] = 1.0 - float(np.clip(similarity, -1.0, 1.0))
            if descriptor.shape[1]:
                previous_descriptor = descriptor[:, -1].copy()
            cosine_distance[start:stop] = block_distance
    else:
        cosine_distance = base_band_cosine_distance

    rms_delta = np.zeros_like(rms)
    if rms.size > 1:
        rms_delta[1:] = np.abs(np.diff(rms))

    cosine_weight, flux_weight, onset_weight, rms_weight = _resolve_novelty_weights(config)
    primary_novelty = (
        cosine_weight * _zscore(cosine_distance)
        + flux_weight * _zscore(spectral_flux)
        + onset_weight * _zscore(onset_strength)
        + rms_weight * _zscore(rms_delta)
    )
    primary_novelty = gaussian_filter1d(primary_novelty.astype(np.float32), sigma=max(config.smooth_sigma, 0.1))
    base_change_curve = (
        0.68 * _zscore(base_band_cosine_distance)
        + 0.32 * _zscore(spectral_flux)
    )
    base_change_curve = gaussian_filter1d(
        base_change_curve.astype(np.float32),
        sigma=max(config.smooth_sigma, 0.1),
    )
    multiscale_bundle = _build_multiscale_change_bundle(
        samples=samples,
        sr=sr,
        target_times=times,
        base_n_fft=n_fft,
        base_hop=hop,
        primary_novelty=primary_novelty,
        config=config,
        base_change_curve=base_change_curve,
    )
    multiscale_weight = float(np.clip(config.multiscale_weight, 0.0, 1.0))
    novelty = (
        (1.0 - multiscale_weight) * _zscore(primary_novelty)
        + multiscale_weight * _zscore(multiscale_bundle["combined"])
    )
    novelty = gaussian_filter1d(novelty.astype(np.float32), sigma=max(config.smooth_sigma * 0.5, 0.1))

    if display_spectrum_sum is None:
        raise ValueError("音频中没有可显示的频谱帧。")
    safe_display_counts = np.maximum(display_bin_counts, 1)
    display_spectrum = np.sqrt(
        np.maximum(display_spectrum_sum / safe_display_counts[None, :], 0.0)
    ).astype(np.float32)
    display_times = (display_time_sum / safe_display_counts).astype(np.float32)
    dynamic_range = max(20.0, float(config.spectrogram_dynamic_range_db))
    display_db = _magnitude_to_relative_db(display_spectrum, dynamic_range)

    return {
        "times": times.astype(np.float32),
        "novelty": novelty.astype(np.float32),
        "rms": rms.astype(np.float32),
        "spectral_centroid_hz": centroid.astype(np.float32),
        "low_band_ratio": low_band_ratio.astype(np.float32),
        "mid_band_ratio": mid_band_ratio.astype(np.float32),
        "band_energy_ratio": band_energy_ratio.astype(np.float32),
        "high_band_ratio": high_band_ratio.astype(np.float32),
        "bandwidth_hz": bandwidth.astype(np.float32),
        "rolloff_hz": rolloff.astype(np.float32),
        "flatness": flatness.astype(np.float32),
        "spectral_flux": spectral_flux.astype(np.float32),
        "onset_strength": onset_strength.astype(np.float32),
        "spectral_entropy": spectral_entropy.astype(np.float32),
        "novelty_short_scale": multiscale_bundle["short"].astype(np.float32),
        "novelty_medium_scale": multiscale_bundle["medium"].astype(np.float32),
        "novelty_long_scale": multiscale_bundle["long"].astype(np.float32),
        "multiscale_consensus": multiscale_bundle["consensus"].astype(np.float32),
        "display_spectrogram_db": display_db.astype(np.float32),
        "display_spectrogram_magnitude": display_spectrum.astype(np.float32),
        "display_times": display_times.astype(np.float32),
        "display_freqs": display_freqs.astype(np.float32),
        "n_fft": int(n_fft),
        "hop_length": int(hop),
    }


def _detect_peaks(
    novelty: np.ndarray,
    times: np.ndarray,
    config: AnalysisConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    if novelty.size == 0:
        empty = np.array([], dtype=np.int32)
        return empty, {"prominences": np.array([], dtype=np.float32)}, np.array([], dtype=np.float32)

    if times.size > 1:
        frame_duration = float(np.median(np.diff(times)))
    else:
        frame_duration = 0.05

    baseline = gaussian_filter1d(novelty, sigma=max(config.smooth_sigma * 4.0, 1.0))
    residual = novelty - baseline
    residual_std = float(np.std(residual))
    robust_scale = float(np.median(np.abs(residual - np.median(residual))) * 1.4826)
    scale = max(residual_std, robust_scale, 0.05)

    threshold = baseline + config.threshold_sigma * scale
    min_distance_frames = max(1, int(config.min_event_distance_sec / max(frame_duration, 1e-4)))
    prominence = max(config.prominence_sigma * scale, 0.05)

    peaks, properties = find_peaks(
        novelty,
        height=threshold,
        distance=min_distance_frames,
        prominence=prominence,
    )

    if peaks.size == 0:
        fallback_height = float(np.quantile(novelty, 0.92))
        peaks, properties = find_peaks(
            novelty,
            height=fallback_height,
            distance=min_distance_frames,
            prominence=0.0,
        )

    return peaks.astype(np.int32), properties, threshold.astype(np.float32)


def _segment_features(samples: np.ndarray, sr: int, config: AnalysisConfig) -> dict[str, float | str]:
    if samples.size == 0:
        samples = np.zeros(max(512, config.n_fft), dtype=np.float32)

    n_fft, hop = _spectrogram_params(samples.size, config)
    spectrum_sum: np.ndarray | None = None
    power_sum: np.ndarray | None = None
    frame_count = 0
    frame_rms_parts: list[np.ndarray] = []
    flux_sum = 0.0
    flux_value_count = 0
    previous_log_spectrum: np.ndarray | None = None
    freqs = np.array([], dtype=np.float32)
    for freqs, _, magnitude in _iter_magnitude_blocks(samples, sr, n_fft, hop):
        safe_magnitude = np.maximum(magnitude, 0.0)
        block_sum = np.sum(safe_magnitude, axis=1, dtype=np.float64)
        block_power_sum = np.sum(np.square(safe_magnitude), axis=1, dtype=np.float64)
        spectrum_sum = block_sum if spectrum_sum is None else spectrum_sum + block_sum
        power_sum = block_power_sum if power_sum is None else power_sum + block_power_sum
        frame_count += magnitude.shape[1]
        frame_rms_parts.append(_framewise_rms_from_magnitude(magnitude, n_fft))

        log_magnitude = np.log1p(safe_magnitude).astype(np.float32)
        if previous_log_spectrum is not None and log_magnitude.shape[1]:
            boundary_diff = np.maximum(log_magnitude[:, 0] - previous_log_spectrum, 0.0)
            flux_sum += float(np.sum(boundary_diff))
            flux_value_count += boundary_diff.size
        if log_magnitude.shape[1] > 1:
            positive_diff = np.maximum(np.diff(log_magnitude, axis=1), 0.0)
            flux_sum += float(np.sum(positive_diff))
            flux_value_count += positive_diff.size
        if log_magnitude.shape[1]:
            previous_log_spectrum = log_magnitude[:, -1].copy()

    if spectrum_sum is None or power_sum is None or frame_count == 0:
        raise ValueError("音频片段中没有可分析的频谱帧。")
    mean_spectrum = np.maximum(spectrum_sum / frame_count, 0.0)
    mean_power = np.maximum(power_sum / frame_count, 0.0)
    total_magnitude = float(np.sum(mean_spectrum))
    total_energy = float(np.sum(mean_power))
    has_signal = total_energy > np.finfo(np.float32).tiny

    low_mask = freqs < 250.0
    mid_mask = (freqs >= 250.0) & (freqs < 2000.0)
    high_mask = freqs >= 2000.0

    if has_signal and total_magnitude > np.finfo(np.float32).tiny:
        centroid = float(np.sum(freqs * mean_spectrum) / total_magnitude)
        bandwidth = float(
            np.sqrt(
                np.sum(np.square(freqs - centroid) * mean_spectrum)
                / total_magnitude
            )
        )
        cumulative_energy = np.cumsum(mean_power) / total_energy
        rolloff_index = min(
            int(np.searchsorted(cumulative_energy, 0.85, side="left")),
            freqs.shape[0] - 1,
        )
        rolloff = float(freqs[rolloff_index])
        safe_power = np.maximum(mean_power, np.finfo(np.float32).tiny)
        flatness = float(np.exp(np.mean(np.log(safe_power))) / np.mean(mean_power))
        distribution = mean_power / total_energy
        spectral_entropy = float(
            -np.sum(
                distribution
                * np.log(np.maximum(distribution, np.finfo(np.float64).tiny))
            )
            / max(math.log(max(2, distribution.size)), 1e-8)
        )
        spectral_crest = float(
            np.max(mean_power) / max(float(np.mean(mean_power)), np.finfo(np.float32).tiny)
        )
    else:
        centroid = 0.0
        bandwidth = 0.0
        rolloff = 0.0
        flatness = 0.0
        spectral_entropy = 0.0
        spectral_crest = 0.0

    frame_rms = np.concatenate(frame_rms_parts).astype(np.float32)
    temporal_variability = float(np.std(frame_rms) / max(float(np.mean(frame_rms)), 1e-10))
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-10))
    dynamic_range_db = float(np.quantile(frame_db, 0.90) - np.quantile(frame_db, 0.10)) if frame_db.size else 0.0
    mean_spectral_flux = flux_sum / flux_value_count if flux_value_count else 0.0

    frequency_step = float(np.median(np.diff(freqs))) if freqs.size > 1 else float(sr)
    minimum_peak_distance = max(1, int(round(50.0 / max(frequency_step, 1e-8))))
    peak_candidates, _ = find_peaks(mean_power, distance=minimum_peak_distance)
    peak_candidates = peak_candidates[freqs[peak_candidates] >= 20.0]
    if has_signal and peak_candidates.size:
        dominant_indices = peak_candidates[
            np.argsort(mean_power[peak_candidates])[-3:][::-1]
        ]
    elif has_signal:
        dominant_indices = np.argsort(mean_power)[-3:][::-1]
    else:
        dominant_indices = np.array([], dtype=np.int64)
    dominant_freqs = ", ".join(f"{freqs[index]:.0f}Hz" for index in dominant_indices)

    return {
        "rms": _safe_rms(samples),
        "centroid_hz": centroid,
        "bandwidth_hz": bandwidth,
        "rolloff_hz": rolloff,
        "flatness": flatness,
        "spectral_entropy": spectral_entropy,
        "spectral_crest": spectral_crest,
        "temporal_variability": temporal_variability,
        "dynamic_range_db": dynamic_range_db,
        "mean_spectral_flux": mean_spectral_flux,
        "low_ratio": float(np.sum(mean_power[low_mask]) / total_energy) if has_signal else 0.0,
        "mid_ratio": float(np.sum(mean_power[mid_mask]) / total_energy) if has_signal else 0.0,
        "high_ratio": float(np.sum(mean_power[high_mask]) / total_energy) if has_signal else 0.0,
        "dominant_freqs": dominant_freqs,
    }


_AGGREGATE_FEATURE_COLUMNS = (
    "rms",
    "spectral_centroid_hz",
    "bandwidth_hz",
    "rolloff_hz",
    "flatness",
    "spectral_entropy",
    "spectral_flux",
    "low_band_ratio",
    "mid_band_ratio",
    "high_band_ratio",
)


def _build_frame_feature_index(feature_table: pd.DataFrame) -> dict[str, np.ndarray]:
    row_count = len(feature_table)
    index = {
        "time_sec": pd.to_numeric(
            feature_table.get("time_sec", pd.Series(dtype=float)),
            errors="coerce",
        ).to_numpy(dtype=np.float64)
    }
    for column in _AGGREGATE_FEATURE_COLUMNS:
        if column in feature_table.columns:
            values = pd.to_numeric(feature_table[column], errors="coerce").to_numpy(
                dtype=np.float32
            )
        else:
            values = np.zeros(row_count, dtype=np.float32)
        index[column] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return index


def _feature_window_bounds(
    times: np.ndarray,
    start_sec: float,
    end_sec: float,
) -> tuple[int, int]:
    if times.size == 0:
        return 0, 0
    left = int(np.searchsorted(times, float(start_sec), side="left"))
    right = int(np.searchsorted(times, float(end_sec), side="left"))
    if right > left:
        return left, right

    midpoint = (float(start_sec) + float(end_sec)) / 2.0
    nearest = int(np.clip(np.searchsorted(times, midpoint), 0, times.size - 1))
    if nearest > 0 and abs(times[nearest - 1] - midpoint) <= abs(times[nearest] - midpoint):
        nearest -= 1
    return nearest, nearest + 1


def _window_mean_spectrum(
    spectrogram_magnitude: np.ndarray,
    spectrogram_times: np.ndarray,
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    if spectrogram_magnitude.size == 0 or spectrogram_times.size == 0:
        return np.array([], dtype=np.float32)

    time_mask = (spectrogram_times >= float(start_sec)) & (
        spectrogram_times < float(end_sec)
    )
    if not np.any(time_mask):
        midpoint = (float(start_sec) + float(end_sec)) / 2.0
        nearest = int(np.argmin(np.abs(spectrogram_times - midpoint)))
        time_mask[nearest] = True

    return np.mean(
        np.maximum(spectrogram_magnitude[:, time_mask], 0.0),
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)


def _dominant_frequencies_from_spectrum(
    mean_spectrum: np.ndarray,
    freqs: np.ndarray,
    limit: int = 3,
) -> str:
    if mean_spectrum.size == 0 or freqs.size != mean_spectrum.size:
        return ""
    power = np.square(np.maximum(mean_spectrum, 0.0))
    if float(np.max(power)) <= np.finfo(np.float32).tiny:
        return ""

    frequency_step = float(np.median(np.diff(freqs))) if freqs.size > 1 else 1.0
    minimum_distance = max(1, int(round(50.0 / max(frequency_step, 1e-8))))
    candidates, _ = find_peaks(power, distance=minimum_distance)
    candidates = candidates[freqs[candidates] >= 20.0]
    if candidates.size:
        selected = candidates[np.argsort(power[candidates])[-limit:][::-1]]
    else:
        selected = np.argsort(power)[-limit:][::-1]
    return ", ".join(f"{freqs[index]:.0f}Hz" for index in selected)


def _aggregate_frame_features(
    feature_index: dict[str, np.ndarray],
    analysis_samples: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    spectrogram_magnitude: np.ndarray,
    spectrogram_times: np.ndarray,
    spectrogram_freqs: np.ndarray,
) -> dict[str, float | str]:
    """Summarize a time window from the already-computed whole-piece features."""
    start = max(0.0, float(start_sec))
    end = max(start, float(end_sec))
    feature_times = feature_index.get("time_sec", np.array([], dtype=np.float64))
    frame_start, frame_end = _feature_window_bounds(feature_times, start, end)
    start_sample = min(analysis_samples.size, max(0, int(math.floor(start * sr))))
    end_sample = min(
        analysis_samples.size,
        max(start_sample, int(math.ceil(end * sr))),
    )
    sample_window = analysis_samples[start_sample:end_sample]

    def mean_column(name: str) -> float:
        values = feature_index.get(name)
        if values is None or frame_end <= frame_start:
            return 0.0
        return float(np.mean(values[frame_start:frame_end], dtype=np.float64))

    all_frame_rms = feature_index.get("rms", np.array([], dtype=np.float64))
    frame_rms = all_frame_rms[frame_start:frame_end]
    rms = _safe_rms(sample_window)
    if frame_rms.size:
        temporal_variability = float(
            np.std(frame_rms) / max(float(np.mean(frame_rms)), 1e-10)
        )
        frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-10))
        dynamic_range_db = float(
            np.quantile(frame_db, 0.90) - np.quantile(frame_db, 0.10)
        )
    else:
        temporal_variability = 0.0
        dynamic_range_db = 0.0

    mean_spectrum = _window_mean_spectrum(
        spectrogram_magnitude,
        spectrogram_times,
        start,
        end,
    )
    if rms > 1e-10 and mean_spectrum.size:
        mean_power = np.square(np.maximum(mean_spectrum, 0.0))
        spectral_crest = float(
            np.max(mean_power)
            / max(float(np.mean(mean_power)), np.finfo(np.float32).tiny)
        )
        dominant_freqs = _dominant_frequencies_from_spectrum(
            mean_spectrum,
            spectrogram_freqs,
        )
    else:
        spectral_crest = 0.0
        dominant_freqs = ""

    return {
        "rms": rms,
        "centroid_hz": mean_column("spectral_centroid_hz"),
        "bandwidth_hz": mean_column("bandwidth_hz"),
        "rolloff_hz": mean_column("rolloff_hz"),
        "flatness": mean_column("flatness"),
        "spectral_entropy": mean_column("spectral_entropy"),
        "spectral_crest": spectral_crest,
        "temporal_variability": temporal_variability,
        "dynamic_range_db": dynamic_range_db,
        "mean_spectral_flux": mean_column("spectral_flux"),
        "low_ratio": mean_column("low_band_ratio"),
        "mid_ratio": mean_column("mid_band_ratio"),
        "high_ratio": mean_column("high_band_ratio"),
        "dominant_freqs": dominant_freqs,
    }


def _channel_bias_from_envelope(
    channel_envelope: dict[str, object],
    center_sec: float,
    window_sec: float,
) -> str:
    channel_count = int(channel_envelope.get("channel_count", 0))
    rms = np.asarray(channel_envelope.get("rms", np.empty((0, 0))), dtype=np.float32)
    times = np.asarray(channel_envelope.get("times", np.array([])), dtype=np.float32)
    if channel_count < 2 or rms.shape[0] < 2:
        return "单声道"
    if times.size == 0 or rms.shape[1] != times.size:
        return "平衡"

    half_window = max(0.5, window_sec) / 2.0
    mask = (times >= center_sec - half_window) & (times <= center_sec + half_window)
    if not bool(np.any(mask)):
        closest = int(np.argmin(np.abs(times - center_sec)))
        mask = np.zeros(times.shape, dtype=bool)
        mask[closest] = True

    left_rms = float(np.sqrt(np.mean(np.square(rms[0, mask]), dtype=np.float64)))
    right_rms = float(np.sqrt(np.mean(np.square(rms[1, mask]), dtype=np.float64)))
    if left_rms > right_rms * 1.15:
        return "左侧偏强"
    if right_rms > left_rms * 1.15:
        return "右侧偏强"
    return "平衡"


def _event_candidate_labels(
    before: dict[str, float | str],
    after: dict[str, float | str],
    is_major_boundary: bool,
) -> list[str]:
    labels = ["新事件出现"]

    before_centroid = max(float(before["centroid_hz"]), 1.0)
    after_centroid = float(after["centroid_hz"])
    before_bandwidth = max(float(before["bandwidth_hz"]), 1.0)
    after_bandwidth = float(after["bandwidth_hz"])

    centroid_ratio = after_centroid / before_centroid
    bandwidth_ratio = after_bandwidth / before_bandwidth
    low_delta = float(after["low_ratio"]) - float(before["low_ratio"])
    high_delta = float(after["high_ratio"]) - float(before["high_ratio"])
    flatness_delta = float(after["flatness"]) - float(before["flatness"])
    rms_ratio = float(after["rms"]) / max(float(before["rms"]), 1e-8)
    rolloff_ratio = float(after["rolloff_hz"]) / max(float(before["rolloff_hz"]), 1.0)

    if centroid_ratio > 1.18 or high_delta > 0.08 or rolloff_ratio > 1.12:
        labels.append("高频扩展")

    if flatness_delta > 0.05 and high_delta > 0.04:
        labels.append("噪声侵入")

    if rms_ratio > 1.28:
        labels.append("能量增强")
    elif rms_ratio < 0.78:
        labels.append("能量减弱")

    if (
        abs(centroid_ratio - 1.0) > 0.15
        or abs(bandwidth_ratio - 1.0) > 0.16
        or abs(flatness_delta) > 0.05
    ):
        labels.append("音色突变")

    if (bandwidth_ratio > 1.15 and rms_ratio > 1.08) or (high_delta > 0.05 and low_delta > -0.02):
        labels.append("材料聚集")

    if rms_ratio < 0.86 and bandwidth_ratio < 0.92 and high_delta < -0.04:
        labels.append("消散")

    if is_major_boundary:
        labels.append(CANDIDATE_BOUNDARY_LABEL)

    return _unique_labels(labels)


def _boundary_state_change_score(
    before: dict[str, float | str],
    after: dict[str, float | str],
) -> tuple[float, bool]:
    before_rms = max(float(before["rms"]), 1e-8)
    after_rms = float(after["rms"])
    before_centroid = max(float(before["centroid_hz"]), 1.0)
    after_centroid = float(after["centroid_hz"])
    before_bandwidth = max(float(before["bandwidth_hz"]), 1.0)
    after_bandwidth = float(after["bandwidth_hz"])
    before_rolloff = max(float(before["rolloff_hz"]), 1.0)
    after_rolloff = float(after["rolloff_hz"])

    rms_shift = abs(math.log2(after_rms / before_rms))
    centroid_shift = abs(after_centroid / before_centroid - 1.0)
    bandwidth_shift = abs(after_bandwidth / before_bandwidth - 1.0)
    rolloff_shift = abs(after_rolloff / before_rolloff - 1.0)
    high_shift = abs(float(after["high_ratio"]) - float(before["high_ratio"]))
    flatness_shift = abs(float(after["flatness"]) - float(before["flatness"]))

    flag_count = sum(
        [
            rms_shift >= 0.30,
            centroid_shift >= 0.15,
            bandwidth_shift >= 0.16,
            rolloff_shift >= 0.12,
            high_shift >= 0.08,
            flatness_shift >= 0.05,
        ]
    )

    score = (
        min(rms_shift / 0.30, 1.6) * 0.24
        + min(centroid_shift / 0.15, 1.6) * 0.20
        + min(bandwidth_shift / 0.16, 1.6) * 0.16
        + min(rolloff_shift / 0.12, 1.6) * 0.14
        + min(high_shift / 0.08, 1.6) * 0.14
        + min(flatness_shift / 0.05, 1.6) * 0.12
    )
    is_clear_change = flag_count >= 2 or score >= 1.05
    return float(score), bool(is_clear_change)


def _section_candidate_labels(
    start_features: dict[str, float | str],
    end_features: dict[str, float | str],
    mean_features: dict[str, float | str],
    event_count: int,
) -> list[str]:
    labels: list[str] = []
    start_rms = max(float(start_features["rms"]), 1e-8)
    end_rms = float(end_features["rms"])
    rms_ratio = end_rms / start_rms
    flatness_delta = float(end_features["flatness"]) - float(start_features["flatness"])
    high_delta = float(end_features["high_ratio"]) - float(start_features["high_ratio"])
    centroid_ratio = float(end_features["centroid_hz"]) / max(float(start_features["centroid_hz"]), 1.0)

    if event_count <= 1 and 0.88 <= rms_ratio <= 1.14 and abs(centroid_ratio - 1.0) <= 0.12:
        labels.append("稳态持续")

    if event_count >= 3 or (rms_ratio > 1.12 and float(mean_features["bandwidth_hz"]) > float(start_features["bandwidth_hz"]) * 1.08):
        labels.append("材料聚集")

    if rms_ratio < 0.84:
        labels.append("消散")
        labels.append("能量减弱")
    elif rms_ratio > 1.18:
        labels.append("能量增强")

    if centroid_ratio > 1.15 or high_delta > 0.07:
        labels.append("高频扩展")

    if flatness_delta > 0.05 and high_delta > 0.04:
        labels.append("噪声侵入")

    entropy = float(mean_features.get("spectral_entropy", 0.0))
    crest = float(mean_features.get("spectral_crest", 0.0))
    temporal_variability = float(mean_features.get("temporal_variability", 0.0))
    if entropy >= 0.82 and float(mean_features["flatness"]) >= 0.12:
        labels.append("频谱扩散")
    elif crest >= 12.0 and entropy <= 0.72:
        labels.append("频谱集中")
    if temporal_variability >= 0.60:
        labels.append("纹理活跃")

    if not labels:
        labels.append("稳态持续" if event_count <= 1 else "材料聚集")

    return _unique_labels(labels)


def _event_summary(labels: list[str], channel_bias: str) -> str:
    summary_bits = labels[:4]
    if channel_bias not in {"平衡", "单声道"}:
        summary_bits.append(f"{channel_bias}声部更突出")
    return "；".join(summary_bits)


def _build_event_table(
    channel_envelope: dict[str, object],
    analysis_samples: np.ndarray,
    sr: int,
    peak_indices: np.ndarray,
    peak_properties: dict[str, np.ndarray],
    novelty: np.ndarray,
    threshold: np.ndarray,
    multiscale_consensus: np.ndarray,
    times: np.ndarray,
    duration_sec: float,
    config: AnalysisConfig,
    feature_index: dict[str, np.ndarray],
    spectrogram_magnitude: np.ndarray,
    spectrogram_times: np.ndarray,
    spectrogram_freqs: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    peak_times = times[peak_indices] if peak_indices.size else np.array([], dtype=np.float32)
    prominences = peak_properties.get("prominences", np.full(peak_indices.shape, np.nan))

    for index, peak_time in enumerate(peak_times):
        context = max(1.0, config.context_window_sec)
        before_start = max(0.0, float(peak_time - context))
        before_end = max(before_start + 0.25, float(peak_time))
        after_start = min(duration_sec, float(peak_time))
        after_end = min(duration_sec, float(peak_time + context))

        before_features = _aggregate_frame_features(
            feature_index,
            analysis_samples,
            sr,
            before_start,
            before_end,
            spectrogram_magnitude,
            spectrogram_times,
            spectrogram_freqs,
        )
        after_features = _aggregate_frame_features(
            feature_index,
            analysis_samples,
            sr,
            after_start,
            after_end,
            spectrogram_magnitude,
            spectrogram_times,
            spectrogram_freqs,
        )
        bias = _channel_bias_from_envelope(
            channel_envelope,
            float(peak_time),
            context,
        )
        boundary_state_change_score, boundary_state_change_flag = _boundary_state_change_score(before_features, after_features)
        peak_index = int(peak_indices[index])
        threshold_value = float(threshold[peak_index]) if peak_index < threshold.size else 0.0
        threshold_excess = float(novelty[peak_index]) - threshold_value
        consensus_value = float(multiscale_consensus[peak_index]) if peak_index < multiscale_consensus.size else 1.0
        prominence_value = float(prominences[index]) if index < len(prominences) else math.nan
        prominence_evidence = 0.0 if not math.isfinite(prominence_value) else 1.0 - math.exp(-max(prominence_value, 0.0) / 0.75)
        excess_evidence = 1.0 - math.exp(-max(threshold_excess, 0.0) / 0.50)
        state_evidence = float(np.clip(boundary_state_change_score / 1.20, 0.0, 1.0))
        evidence_score = (
            0.25 * excess_evidence
            + 0.20 * prominence_evidence
            + 0.35 * state_evidence
            + 0.20 * consensus_value
        )
        if evidence_score >= 0.72:
            evidence_label = "高证据"
        elif evidence_score >= 0.46:
            evidence_label = "中证据"
        else:
            evidence_label = "低证据"

        rows.append(
            {
                "event_id": index + 1,
                "time_sec": round(float(peak_time), 3),
                "time_label": format_seconds(float(peak_time)),
                "strength": round(float(novelty[peak_index]), 4),
                "prominence": round(prominence_value, 4) if math.isfinite(prominence_value) else math.nan,
                "threshold_value": round(threshold_value, 4),
                "threshold_excess": round(threshold_excess, 4),
                "multiscale_consensus": round(consensus_value, 4),
                "state_change_score": round(boundary_state_change_score, 4),
                "evidence_score": round(evidence_score, 4),
                "evidence_label": evidence_label,
                "evidence_summary": (
                    f"多尺度 {int(round(consensus_value * 3))}/3；"
                    f"阈值超出 {threshold_excess:.2f}；状态变化 {boundary_state_change_score:.2f}"
                ),
                "channel_bias": bias,
                "pre_rms": round(float(before_features["rms"]), 6),
                "pre_centroid_hz": round(float(before_features["centroid_hz"]), 1),
                "pre_high_ratio": round(float(before_features["high_ratio"]), 4),
                "pre_flatness": round(float(before_features["flatness"]), 4),
                "post_rms": round(float(after_features["rms"]), 6),
                "post_centroid_hz": round(float(after_features["centroid_hz"]), 1),
                "post_bandwidth_hz": round(float(after_features["bandwidth_hz"]), 1),
                "post_rolloff_hz": round(float(after_features["rolloff_hz"]), 1),
                "post_low_ratio": round(float(after_features["low_ratio"]), 4),
                "post_mid_ratio": round(float(after_features["mid_ratio"]), 4),
                "post_high_ratio": round(float(after_features["high_ratio"]), 4),
                "post_flatness": round(float(after_features["flatness"]), 4),
                "dominant_freqs": str(after_features["dominant_freqs"]),
                "_boundary_state_change_score": round(boundary_state_change_score, 4),
                "_boundary_state_change_flag": bool(boundary_state_change_flag),
                "_before_features": before_features,
                "_after_features": after_features,
            }
        )

    event_table = pd.DataFrame(rows)
    if event_table.empty:
        return event_table

    event_table["boundary_score"] = (
        0.55 * np.clip(event_table["state_change_score"].astype(float) / 1.20, 0.0, 1.0)
        + 0.30 * event_table["evidence_score"].astype(float)
        + 0.15 * event_table["multiscale_consensus"].astype(float)
    ).round(4)
    event_table["is_major_boundary"] = (
        event_table["_boundary_state_change_flag"].astype(bool)
        & (event_table["boundary_score"] >= 0.52)
        & (event_table["multiscale_consensus"] >= (1.0 / 3.0))
    )
    if not bool(event_table["is_major_boundary"].any()):
        candidate_pool = event_table.loc[event_table["_boundary_state_change_flag"]].copy()
        if not candidate_pool.empty:
            event_table.loc[candidate_pool["boundary_score"].idxmax(), "is_major_boundary"] = True

    auto_label_texts: list[str] = []
    summaries: list[str] = []
    for _, row in event_table.iterrows():
        labels = _event_candidate_labels(
            before=row["_before_features"],
            after=row["_after_features"],
            is_major_boundary=bool(row["is_major_boundary"]),
        )
        auto_label_texts.append(join_labels(labels))
        summaries.append(_event_summary(labels, str(row["channel_bias"])))

    event_table["auto_labels"] = auto_label_texts
    event_table["descriptor"] = summaries
    return event_table.drop(columns=["_before_features", "_after_features", "_boundary_state_change_score", "_boundary_state_change_flag"])


def _build_section_table(
    analysis_samples: np.ndarray,
    sr: int,
    duration_sec: float,
    event_table: pd.DataFrame,
    feature_index: dict[str, np.ndarray],
    spectrogram_magnitude: np.ndarray,
    spectrogram_times: np.ndarray,
    spectrogram_freqs: np.ndarray,
) -> pd.DataFrame:
    if event_table.empty:
        boundaries = [0.0, duration_sec]
    else:
        major_boundaries = event_table.loc[event_table["is_major_boundary"], "time_sec"].tolist()
        boundaries = [0.0, *major_boundaries, duration_sec]

    cleaned_boundaries = [boundaries[0]]
    for boundary in boundaries[1:]:
        if boundary - cleaned_boundaries[-1] >= 0.5:
            cleaned_boundaries.append(boundary)
    if cleaned_boundaries[-1] != duration_sec:
        cleaned_boundaries.append(duration_sec)

    rows: list[dict[str, float | int | str]] = []
    event_times = event_table["time_sec"].to_numpy(dtype=float) if not event_table.empty else np.array([], dtype=float)
    for index in range(len(cleaned_boundaries) - 1):
        start = float(cleaned_boundaries[index])
        end = float(cleaned_boundaries[index + 1])
        duration = end - start
        probe_duration = max(0.6, min(duration * 0.3, 2.5))
        mean_features = _aggregate_frame_features(
            feature_index,
            analysis_samples,
            sr,
            start,
            end,
            spectrogram_magnitude,
            spectrogram_times,
            spectrogram_freqs,
        )
        start_features = _aggregate_frame_features(
            feature_index,
            analysis_samples,
            sr,
            start,
            min(end, start + probe_duration),
            spectrogram_magnitude,
            spectrogram_times,
            spectrogram_freqs,
        )
        end_features = _aggregate_frame_features(
            feature_index,
            analysis_samples,
            sr,
            max(start, end - probe_duration),
            end,
            spectrogram_magnitude,
            spectrogram_times,
            spectrogram_freqs,
        )
        event_count = int(np.sum((event_times >= start) & (event_times < end)))
        labels = _section_candidate_labels(start_features, end_features, mean_features, event_count)

        rows.append(
            {
                "section_id": index + 1,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "start_label": format_seconds(start),
                "end_label": format_seconds(end),
                "duration_sec": round(end - start, 3),
                "event_count": event_count,
                "auto_labels": join_labels(labels),
                "descriptor": "；".join(labels[:4]),
                "centroid_hz": round(float(mean_features["centroid_hz"]), 1),
                "rms": round(float(mean_features["rms"]), 6),
                "bandwidth_hz": round(float(mean_features["bandwidth_hz"]), 1),
                "rolloff_hz": round(float(mean_features["rolloff_hz"]), 1),
                "low_ratio": round(float(mean_features["low_ratio"]), 4),
                "mid_ratio": round(float(mean_features["mid_ratio"]), 4),
                "high_ratio": round(float(mean_features["high_ratio"]), 4),
                "flatness": round(float(mean_features["flatness"]), 4),
                "spectral_entropy": round(float(mean_features["spectral_entropy"]), 4),
                "spectral_crest": round(float(mean_features["spectral_crest"]), 4),
                "temporal_variability": round(float(mean_features["temporal_variability"]), 4),
                "dynamic_range_db": round(float(mean_features["dynamic_range_db"]), 2),
                "mean_spectral_flux": round(float(mean_features["mean_spectral_flux"]), 6),
                "event_density_per_min": round(event_count / max(duration / 60.0, 1e-8), 3),
                "dominant_freqs": str(mean_features["dominant_freqs"]),
            }
        )

    return pd.DataFrame(rows)


_STATE_FINGERPRINT_FEATURES = {
    "rms": "energy",
    "centroid_hz": "brightness",
    "bandwidth_hz": "spectral_width",
    "flatness": "noise_character",
    "spectral_entropy": "spectral_dispersion",
    "high_ratio": "high_frequency_presence",
    "temporal_variability": "temporal_variability",
    "mean_spectral_flux": "internal_change",
    "event_density_per_min": "event_density",
}


def _robust_standardize(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    median = float(numeric.median())
    mad = float(np.median(np.abs(numeric.to_numpy(dtype=float) - median)) * 1.4826)
    if mad < 1e-8:
        std = float(numeric.std(ddof=0))
        mad = std if std >= 1e-8 else 1.0
    return ((numeric - median) / mad).clip(-4.0, 4.0)


def _soundscape_state_label(row: pd.Series) -> str:
    if row["noise_character_z"] >= 0.65 and row["spectral_dispersion_z"] >= 0.45:
        return "噪声化扩散"
    if row["energy_z"] >= 0.55 and row["spectral_width_z"] >= 0.45:
        return "高密度聚集"
    if row["energy_z"] <= -0.55 and row["temporal_variability_z"] <= 0.10:
        return "低能量稳态"
    if row["brightness_z"] >= 0.65 and row["high_frequency_presence_z"] >= 0.45:
        return "高频明亮"
    if row["internal_change_z"] >= 0.65 or row["event_density_z"] >= 0.65:
        return "纹理活跃"
    if row["spectral_dispersion_z"] <= -0.55 and row["spectral_width_z"] <= -0.35:
        return "频谱集中"
    return "混合过渡"


def _build_section_fingerprint_table(section_table: pd.DataFrame) -> pd.DataFrame:
    identity_columns = [
        "section_id",
        "start_sec",
        "end_sec",
        "start_label",
        "end_label",
        "duration_sec",
        "auto_labels",
    ]
    if section_table.empty:
        z_columns = [f"{label}_z" for label in _STATE_FINGERPRINT_FEATURES.values()]
        return pd.DataFrame(columns=[*identity_columns, "state_label", *z_columns])

    fingerprint = section_table[identity_columns].copy()
    for source_column, output_label in _STATE_FINGERPRINT_FEATURES.items():
        raw_values = pd.to_numeric(section_table[source_column], errors="coerce").fillna(0.0).astype(float)
        fingerprint[output_label] = raw_values.to_numpy()
        fingerprint[f"{output_label}_z"] = _robust_standardize(raw_values).round(4).to_numpy()
    fingerprint["state_label"] = fingerprint.apply(_soundscape_state_label, axis=1)
    ordered_columns = [
        *identity_columns,
        "state_label",
        *[label for label in _STATE_FINGERPRINT_FEATURES.values()],
        *[f"{label}_z" for label in _STATE_FINGERPRINT_FEATURES.values()],
    ]
    return fingerprint[ordered_columns]


def _build_state_similarity_table(fingerprint_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_section_id",
        "source_state",
        "match_section_id",
        "match_state",
        "similarity_score",
        "is_recurrence_candidate",
    ]
    if len(fingerprint_table) < 2:
        return pd.DataFrame(columns=columns)

    z_columns = [f"{label}_z" for label in _STATE_FINGERPRINT_FEATURES.values()]
    vectors = fingerprint_table[z_columns].to_numpy(dtype=np.float64)
    feature_count = max(1, vectors.shape[1])
    rows: list[dict[str, object]] = []
    for left_index in range(len(fingerprint_table) - 1):
        for right_index in range(left_index + 1, len(fingerprint_table)):
            distance = float(np.linalg.norm(vectors[left_index] - vectors[right_index]) / math.sqrt(feature_count))
            similarity = math.exp(-distance)
            left = fingerprint_table.iloc[left_index]
            right = fingerprint_table.iloc[right_index]
            rows.append(
                {
                    "source_section_id": int(left["section_id"]),
                    "source_state": str(left["state_label"]),
                    "match_section_id": int(right["section_id"]),
                    "match_state": str(right["state_label"]),
                    "similarity_score": round(similarity, 4),
                    "is_recurrence_candidate": bool(similarity >= 0.78),
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values("similarity_score", ascending=False).reset_index(drop=True)


def _build_analysis_metadata(
    source: str | Path | BinaryIO,
    selected_audio: np.ndarray,
    sr: int,
    config: AnalysisConfig,
    source_sha256: str | None = None,
) -> dict[str, object]:
    normalized_audio = np.ascontiguousarray(selected_audio, dtype="<f4")
    audio_digest = hashlib.sha256()
    audio_view = memoryview(normalized_audio).cast("B")
    for start in range(0, len(audio_view), 1024 * 1024):
        audio_digest.update(audio_view[start : start + 1024 * 1024])
    analysis_audio_sha256 = audio_digest.hexdigest()
    return {
        "method_version": ANALYSIS_METHOD_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_name": _resolve_source_name(source),
        "source_sha256": source_sha256 or _source_sha256(source),
        "analysis_audio_sha256": analysis_audio_sha256,
        "audio_sha256": analysis_audio_sha256,
        "sample_rate_hz": int(sr),
        "analysis_principle": "multi-scale spectral change + pre/post sound-state comparison",
        "spectrogram_unit": "relative dB",
        "spectrogram_dynamic_range_db": float(config.spectrogram_dynamic_range_db),
        "rms_definition": "window-normalized frame RMS recovered from the one-sided STFT via Parseval's theorem",
        "spectral_weighting": "magnitude-weighted centroid/bandwidth; power-weighted band ratios, rolloff, flatness, entropy, and crest",
        "silence_policy": "silent frames receive zero-valued spectral descriptors and remain at the spectrogram display floor",
        "window_summary": "event and section descriptors aggregate the whole-piece frame grid without recomputing local STFTs",
        "display_spectrogram_pooling": "peak pooling across frequency and RMS pooling across time to preserve narrow partials and transients",
        "resampling_method": "phase-aligned polyphase resampling on a continuous global sample grid",
        "evidence_score_note": "Evidence score is an interpretable rule-based score, not a statistical probability.",
    }


def _source_sha256(source: str | Path | BinaryIO) -> str:
    digest = hashlib.sha256()
    if isinstance(source, (str, Path)):
        with Path(source).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    position = source.tell() if hasattr(source, "tell") else None
    try:
        source.seek(0)
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    finally:
        if position is not None:
            source.seek(position)
    return digest.hexdigest()


def analyze_audio(
    source: str | Path | BinaryIO,
    config: AnalysisConfig | None = None,
    channel_mode: ChannelMode = "mix",
    source_sha256: str | None = None,
) -> dict[str, object]:
    analysis_config = config or AnalysisConfig()
    audio, selected, sr, channel_envelope = _load_analysis_audio(
        source,
        analysis_config.target_sr,
        channel_mode,
    )
    if sr <= 0 or selected.size < 2:
        raise ValueError("音频为空或过短，无法进行频谱分析。")
    source_channel_count = int(channel_envelope.get("channel_count", audio.shape[0]))
    del audio
    duration_sec = float(selected.shape[0] / sr)

    novelty_bundle = _build_novelty_curve(selected, sr, analysis_config)
    peak_indices, peak_properties, threshold = _detect_peaks(
        novelty_bundle["novelty"],
        novelty_bundle["times"],
        analysis_config,
    )

    feature_table = _build_feature_table(
        times=novelty_bundle["times"],
        rms=novelty_bundle["rms"],
        centroid=novelty_bundle["spectral_centroid_hz"],
        low_band_ratio=novelty_bundle["low_band_ratio"],
        mid_band_ratio=novelty_bundle["mid_band_ratio"],
        band_energy_ratio=novelty_bundle["band_energy_ratio"],
        high_band_ratio=novelty_bundle["high_band_ratio"],
        bandwidth=novelty_bundle["bandwidth_hz"],
        rolloff=novelty_bundle["rolloff_hz"],
        flatness=novelty_bundle["flatness"],
        spectral_flux=novelty_bundle["spectral_flux"],
        onset_strength=novelty_bundle["onset_strength"],
        spectral_entropy=novelty_bundle["spectral_entropy"],
        novelty_short_scale=novelty_bundle["novelty_short_scale"],
        novelty_medium_scale=novelty_bundle["novelty_medium_scale"],
        novelty_long_scale=novelty_bundle["novelty_long_scale"],
        multiscale_consensus=novelty_bundle["multiscale_consensus"],
        novelty=novelty_bundle["novelty"],
        threshold=threshold,
    )
    feature_index = _build_frame_feature_index(feature_table)

    event_table = _build_event_table(
        channel_envelope=channel_envelope,
        analysis_samples=selected,
        sr=sr,
        peak_indices=peak_indices,
        peak_properties=peak_properties,
        novelty=novelty_bundle["novelty"],
        threshold=threshold,
        multiscale_consensus=novelty_bundle["multiscale_consensus"],
        times=novelty_bundle["times"],
        duration_sec=duration_sec,
        config=analysis_config,
        feature_index=feature_index,
        spectrogram_magnitude=novelty_bundle["display_spectrogram_magnitude"],
        spectrogram_times=novelty_bundle["display_times"],
        spectrogram_freqs=novelty_bundle["display_freqs"],
    )
    section_table = _build_section_table(
        analysis_samples=selected,
        sr=sr,
        duration_sec=duration_sec,
        event_table=event_table,
        feature_index=feature_index,
        spectrogram_magnitude=novelty_bundle["display_spectrogram_magnitude"],
        spectrogram_times=novelty_bundle["display_times"],
        spectrogram_freqs=novelty_bundle["display_freqs"],
    )
    section_fingerprint_table = _build_section_fingerprint_table(section_table)
    state_similarity_table = _build_state_similarity_table(section_fingerprint_table)
    analysis_metadata = _build_analysis_metadata(
        source,
        selected,
        sr,
        analysis_config,
        source_sha256=source_sha256,
    )
    analysis_metadata["source_channel_count"] = source_channel_count
    analysis_metadata["retained_analysis_channel_count"] = 1
    analysis_metadata["long_form_processing"] = "continuous blockwise STFT on one global timeline"

    summary_lines: list[str] = []
    if not event_table.empty:
        summary_lines.append(
            f"共检测到 {len(event_table)} 个疑似频谱新事件，其中 {int(event_table['is_major_boundary'].sum())} 个被标记为候选边界。"
        )
        high_evidence_count = int((event_table["evidence_label"] == "高证据").sum())
        medium_evidence_count = int((event_table["evidence_label"] == "中证据").sum())
        summary_lines.append(
            f"事件证据分层：高证据 {high_evidence_count} 个，中证据 {medium_evidence_count} 个；证据分数不是统计概率。"
        )
        for row in event_table.loc[event_table["is_major_boundary"]].itertuples():
            summary_lines.append(f"{row.time_label}: {row.descriptor}（{row.evidence_label}，分数 {row.evidence_score:.2f}）")
    else:
        summary_lines.append("未检测到显著的新频谱事件，建议降低阈值或缩短最小事件间隔后再试。")

    recurrence_count = int(state_similarity_table["is_recurrence_candidate"].sum()) if not state_similarity_table.empty else 0
    summary_lines.append(
        f"共形成 {len(section_fingerprint_table)} 个声景状态指纹，发现 {recurrence_count} 对状态再现候选。"
    )

    return {
        "config": analysis_config.to_dict(),
        "analysis_metadata": analysis_metadata,
        "audio": selected.reshape(1, -1),
        "selected_audio": selected,
        "sr": sr,
        "duration_sec": duration_sec,
        "channel_mode": channel_mode,
        "times": novelty_bundle["times"],
        "spectrogram_db": novelty_bundle["display_spectrogram_db"],
        "spectrogram_times": novelty_bundle["display_times"],
        "spectrogram_freqs": novelty_bundle["display_freqs"],
        "feature_table": feature_table,
        "novelty": novelty_bundle["novelty"],
        "threshold": threshold,
        "peak_indices": peak_indices,
        "peak_times": novelty_bundle["times"][peak_indices] if peak_indices.size else np.array([], dtype=np.float32),
        "hop_length": novelty_bundle["hop_length"],
        "event_table": event_table,
        "section_table": section_table,
        "section_fingerprint_table": section_fingerprint_table,
        "state_similarity_table": state_similarity_table,
        "summary_lines": summary_lines,
        "event_label_vocab": EVENT_LABEL_VOCAB,
        "section_label_vocab": SECTION_LABEL_VOCAB,
        "feature_column_labels": FEATURE_COLUMN_LABELS,
    }
