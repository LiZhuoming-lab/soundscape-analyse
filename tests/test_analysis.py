from __future__ import annotations

"""Regression tests for the Soundscape Analyse audio-analysis pipeline."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import soundfile as sf

from spectral_tool.assistant import (
    annotate_event_interaction_levels,
    annotate_event_similarity_groups,
    build_assistant_overlay_component,
    build_event_assistant_payload,
)
from spectral_tool.analysis import (
    CANDIDATE_BOUNDARY_LABEL,
    AnalysisConfig,
    _aggregate_frequency_bins,
    _aggregate_frequency_peaks,
    _build_multiscale_change_bundle,
    _boundary_state_change_score,
    _framewise_band_energy_ratios,
    _framewise_centroid_and_bandwidth,
    _framewise_flatness,
    _framewise_rms_from_magnitude,
    _framewise_rolloff,
    _framewise_spectral_entropy,
    _iter_magnitude_blocks,
    _load_analysis_audio,
    _load_audio,
    _read_selected_soundfile,
    _resolve_novelty_weights,
    _segment_features,
    _magnitude_to_relative_db,
    analyze_audio,
    build_audio_excerpt_wav,
)
from spectral_tool.state.audio_state import build_analysis_signature
from spectral_tool.visualization import (
    _downsample_waveform_extrema,
    _prepare_feature_curve_data,
    build_interactive_novelty_chart,
    build_local_waveform_chart,
    build_novelty_threshold_chart,
    build_synced_overview_player_html,
    build_synced_waveform_player_html,
    plot_local_spectrogram,
    plot_local_waveform,
    plot_spectrogram,
)


class AnalysisTestCase(unittest.TestCase):
    def test_silent_frames_have_zero_descriptors_and_dark_spectrogram(self) -> None:
        freqs = np.linspace(0.0, 8000.0, 9, dtype=np.float32)
        magnitude = np.zeros((9, 4), dtype=np.float32)

        centroid, bandwidth = _framewise_centroid_and_bandwidth(freqs, magnitude)
        low, mid, high, ratio = _framewise_band_energy_ratios(freqs, magnitude)

        for values in [
            centroid,
            bandwidth,
            low,
            mid,
            high,
            ratio,
            _framewise_rolloff(freqs, magnitude),
            _framewise_flatness(magnitude),
            _framewise_spectral_entropy(magnitude),
        ]:
            np.testing.assert_array_equal(values, np.zeros(values.shape, dtype=values.dtype))

        relative_db = _magnitude_to_relative_db(magnitude, 90.0)
        np.testing.assert_array_equal(relative_db, np.full(magnitude.shape, -90.0, dtype=np.float32))

    def test_band_energy_ratios_use_power_spectrum(self) -> None:
        freqs = np.array([100.0, 1000.0, 3000.0], dtype=np.float32)
        magnitude = np.array([[1.0], [1.0], [2.0]], dtype=np.float32)

        low, mid, high, ratio = _framewise_band_energy_ratios(freqs, magnitude)

        self.assertAlmostEqual(float(low[0]), 1.0 / 6.0, places=6)
        self.assertAlmostEqual(float(mid[0]), 1.0 / 6.0, places=6)
        self.assertAlmostEqual(float(high[0]), 4.0 / 6.0, places=6)
        self.assertAlmostEqual(float(ratio[0]), 2.0, places=6)

    def test_stft_rms_matches_time_domain_amplitude_across_fft_sizes(self) -> None:
        sr = 22050
        amplitude = 0.20
        time_axis = np.arange(sr, dtype=np.float32) / sr
        samples = (amplitude * np.sin(2 * np.pi * 440 * time_axis)).astype(np.float32)
        measured: list[float] = []

        for n_fft, hop in [(1024, 256), (4096, 1024)]:
            block_values: list[np.ndarray] = []
            for _, _, magnitude in _iter_magnitude_blocks(samples, sr, n_fft, hop):
                block_values.append(_framewise_rms_from_magnitude(magnitude, n_fft))
            measured.append(float(np.median(np.concatenate(block_values))))

        expected = amplitude / np.sqrt(2.0)
        self.assertAlmostEqual(measured[0], expected, delta=0.002)
        self.assertAlmostEqual(measured[1], expected, delta=0.002)
        self.assertAlmostEqual(measured[0], measured[1], delta=0.001)

    def test_dominant_frequencies_represent_separate_spectral_peaks(self) -> None:
        sr = 22050
        time_axis = np.arange(sr * 2, dtype=np.float32) / sr
        samples = (
            0.30 * np.sin(2 * np.pi * 220 * time_axis)
            + 0.20 * np.sin(2 * np.pi * 880 * time_axis)
            + 0.12 * np.sin(2 * np.pi * 1600 * time_axis)
        ).astype(np.float32)

        features = _segment_features(samples, sr, AnalysisConfig(n_fft=4096, hop_length=1024))
        dominant = [
            float(value.removesuffix("Hz"))
            for value in str(features["dominant_freqs"]).split(", ")
        ]

        self.assertEqual(len(dominant), 3)
        self.assertTrue(any(abs(value - 220.0) < 12.0 for value in dominant))
        self.assertTrue(any(abs(value - 880.0) < 12.0 for value in dominant))
        self.assertTrue(any(abs(value - 1600.0) < 12.0 for value in dominant))

    def test_frequency_bin_aggregation_matches_array_split_reference(self) -> None:
        rng = np.random.default_rng(41)
        freqs = np.linspace(0.0, 22050.0, 17, dtype=np.float32)
        spectrum = rng.random((17, 9), dtype=np.float32)
        groups = np.array_split(np.arange(spectrum.shape[0]), 5)

        aggregated_freqs, aggregated_spectrum = _aggregate_frequency_bins(
            freqs,
            spectrum,
            group_count=5,
        )

        expected_freqs = np.array([np.mean(freqs[group]) for group in groups], dtype=np.float32)
        expected_spectrum = np.stack(
            [np.mean(spectrum[group], axis=0) for group in groups],
            axis=0,
        ).astype(np.float32)
        np.testing.assert_allclose(aggregated_freqs, expected_freqs, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(aggregated_spectrum, expected_spectrum, rtol=1e-6, atol=1e-6)

    def test_frequency_peak_pooling_preserves_narrow_partials(self) -> None:
        freqs = np.linspace(0.0, 7000.0, 8, dtype=np.float32)
        spectrum = np.zeros((8, 2), dtype=np.float32)
        spectrum[2, 0] = 0.9
        spectrum[6, 1] = 0.7

        _, pooled = _aggregate_frequency_peaks(freqs, spectrum, group_count=4)

        self.assertEqual(pooled.shape, (4, 2))
        self.assertAlmostEqual(float(np.max(pooled[:, 0])), 0.9, places=6)
        self.assertAlmostEqual(float(np.max(pooled[:, 1])), 0.7, places=6)

    def test_multiscale_bundle_reuses_the_base_scale_curve(self) -> None:
        target_times = np.linspace(0.0, 4.0, 40, dtype=np.float32)
        base_curve = np.sin(target_times).astype(np.float32)
        config = AnalysisConfig(multiscale_enabled=True)

        with mock.patch(
            "spectral_tool.analysis._single_scale_change_curve",
            return_value=np.linspace(-1.0, 1.0, target_times.size, dtype=np.float32),
        ) as scale_builder:
            bundle = _build_multiscale_change_bundle(
                samples=np.zeros(22050, dtype=np.float32),
                sr=22050,
                target_times=target_times,
                base_n_fft=2048,
                base_hop=512,
                primary_novelty=base_curve,
                config=config,
                base_change_curve=base_curve,
            )

        self.assertEqual(scale_builder.call_count, 2)
        self.assertEqual(bundle["medium"].shape, target_times.shape)

    def test_chunked_resampling_matches_whole_file_across_block_boundary(self) -> None:
        source_sr = 44100
        target_sr = 32000
        sample_count = 1_048_577
        time_axis = np.arange(sample_count, dtype=np.float32) / source_sr
        audio = (
            0.10 * np.sin(2 * np.pi * 311 * time_axis)
            + 0.04 * np.sin(2 * np.pi * 2203 * time_axis)
        ).astype(np.float32)
        audio[1_048_566] = 0.8

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "cross-block.wav"
            sf.write(path, audio, source_sr, subtype="FLOAT")
            chunked, actual_sr, _ = _read_selected_soundfile(path, "mix", target_sr)

        from scipy.signal import resample_poly

        whole_file = resample_poly(audio, 320, 441).astype(np.float32)
        self.assertEqual(actual_sr, target_sr)
        self.assertEqual(chunked.shape, whole_file.shape)
        np.testing.assert_allclose(chunked, whole_file, rtol=0.0, atol=1e-6)

    def test_waveform_downsampling_preserves_isolated_transients(self) -> None:
        samples = np.zeros(500_000, dtype=np.float32)
        samples[123_457] = 1.0
        samples[321_123] = -0.75

        times, plotted = _downsample_waveform_extrema(samples, sr=32000, max_bins=2_000)

        self.assertLessEqual(plotted.size, 4_002)
        self.assertEqual(float(np.max(plotted)), 1.0)
        self.assertEqual(float(np.min(plotted)), -0.75)
        self.assertTrue(np.all(np.diff(times) >= 0.0))

    def test_feature_curve_display_data_is_finite_and_bounded(self) -> None:
        row_count = 20_000
        feature_table = pd.DataFrame(
            {
                "time_sec": np.linspace(0.0, 600.0, row_count),
                "rms": np.linspace(0.0, 1.0, row_count),
                "novelty": np.sin(np.linspace(0.0, 30.0, row_count)),
            }
        )
        feature_table.loc[10, "novelty"] = np.inf

        prepared = _prepare_feature_curve_data(
            feature_table,
            ["rms", "novelty"],
            {"rms": "RMS", "novelty": "Novelty"},
            normalize=True,
            max_time_points=1_000,
        )

        self.assertLessEqual(len(prepared), 2_000)
        self.assertTrue(np.isfinite(prepared["time_sec"]).all())
        self.assertTrue(np.isfinite(prepared["value"]).all())

    def test_spectrogram_uses_stable_event_ids_as_labels(self) -> None:
        spectrogram_db = np.full((8, 20), -24.0, dtype=np.float32)
        times = np.linspace(0.0, 4.0, 20, dtype=np.float32)
        freqs = np.linspace(0.0, 8000.0, 8, dtype=np.float32)

        figure = plot_spectrogram(
            spectrogram_db,
            times,
            freqs,
            np.array([0.5, 1.5, 3.0], dtype=np.float32),
            np.array([4, 9, 15]),
        )

        self.assertEqual([text.get_text() for text in figure.axes[0].texts], ["4", "9", "15"])

        log_figure = plot_spectrogram(
            spectrogram_db,
            times,
            freqs,
            np.array([], dtype=np.float32),
            frequency_scale="log",
        )
        self.assertEqual(log_figure.axes[0].get_yscale(), "log")

    def test_event_charts_separate_detection_curves_from_interactive_points(self) -> None:
        feature_table = pd.DataFrame(
            {
                "time_sec": np.linspace(0.0, 4.0, 20),
                "novelty": np.linspace(0.0, 2.0, 20),
                "threshold": np.full(20, 1.0),
            }
        )
        event_table = pd.DataFrame(
            {
                "event_id": [1],
                "time_sec": [1.0],
                "time_label": ["00:01.00"],
                "manual_time_sec": [1.25],
                "manual_time_label": ["00:01.25"],
                "strength": [1.4],
                "prominence": [0.5],
                "is_major_boundary": [True],
                "auto_labels": [CANDIDATE_BOUNDARY_LABEL],
            }
        )

        point_spec = build_interactive_novelty_chart(feature_table, event_table).to_dict()
        curve_spec = build_novelty_threshold_chart(feature_table).to_dict()
        point_serialized = str(point_spec)
        curve_serialized = str(curve_spec)

        self.assertNotIn("layer", point_spec)
        self.assertIn("display_time_sec", point_serialized)
        self.assertIn("diamond", point_serialized)
        self.assertIn("检测阈值", curve_serialized)

    def test_analysis_signature_accepts_a_precomputed_source_digest(self) -> None:
        with mock.patch(
            "spectral_tool.state.audio_state._source_sha256",
            side_effect=AssertionError("source should not be read"),
        ):
            signature = build_analysis_signature(
                io.BytesIO(b"not-read"),
                AnalysisConfig(),
                "mix",
                audio_sha256="known-digest",
            )

        self.assertEqual(len(signature), 64)

    def test_analysis_audio_falls_back_only_for_decoder_errors(self) -> None:
        fallback_audio = np.zeros((1, 8000), dtype=np.float32)
        with (
            mock.patch(
                "spectral_tool.analysis._read_selected_soundfile",
                side_effect=sf.LibsndfileError(1, "unsupported: "),
            ),
            mock.patch(
                "spectral_tool.analysis._load_audio",
                return_value=(fallback_audio, 8000),
            ) as fallback_loader,
        ):
            audio, selected, sr, _ = _load_analysis_audio(io.BytesIO(b"audio"), None, "mix")

        fallback_loader.assert_called_once()
        self.assertEqual(sr, 8000)
        self.assertEqual(audio.shape, (1, 8000))
        self.assertEqual(selected.shape, (8000,))

        with (
            mock.patch(
                "spectral_tool.analysis._read_selected_soundfile",
                side_effect=ValueError("internal analysis failure"),
            ),
            mock.patch("spectral_tool.analysis._load_audio") as fallback_loader,
        ):
            with self.assertRaisesRegex(ValueError, "internal analysis failure"):
                _load_analysis_audio(io.BytesIO(b"audio"), None, "mix")
        fallback_loader.assert_not_called()

    def test_detects_major_spectral_boundaries(self) -> None:
        rng = np.random.default_rng(7)
        sr = 22050
        segment_duration = 2.0
        sample_count = int(sr * segment_duration)
        time_axis = np.linspace(0.0, segment_duration, sample_count, endpoint=False)

        segment_1 = 0.03 * rng.normal(size=sample_count)
        segment_2 = 0.18 * np.sin(2 * np.pi * 220 * time_axis)
        segment_3 = 0.05 * rng.normal(size=sample_count) + 0.08 * np.sin(2 * np.pi * 1600 * time_axis)
        segment_4 = 0.14 * np.sin(2 * np.pi * 60 * time_axis) + 0.04 * rng.normal(size=sample_count)

        audio = np.concatenate([segment_1, segment_2, segment_3, segment_4]).astype(np.float32)

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "synthetic.wav"
            sf.write(path, audio, sr)

            result = analyze_audio(
                path,
                config=AnalysisConfig(
                    target_sr=22050,
                    n_fft=2048,
                    hop_length=512,
                    smooth_sigma=1.0,
                    threshold_sigma=0.5,
                    prominence_sigma=0.4,
                    min_event_distance_sec=1.0,
                    context_window_sec=1.0,
                ),
            )

        event_times = result["event_table"]["time_sec"].to_numpy()
        self.assertGreaterEqual(len(event_times), 3)
        self.assertTrue(np.any(np.abs(event_times - 2.0) < 0.6))
        self.assertTrue(np.any(np.abs(event_times - 4.0) < 0.6))
        self.assertTrue(np.any(np.abs(event_times - 6.0) < 0.6))
        self.assertIn("feature_table", result)
        self.assertTrue(
            {
                "rms",
                "spectral_centroid_hz",
                "low_band_ratio",
                "mid_band_ratio",
                "band_energy_ratio",
                "high_band_ratio",
                "rolloff_hz",
                "flatness",
                "spectral_flux",
                "onset_strength",
                "spectral_entropy",
                "novelty_short_scale",
                "novelty_medium_scale",
                "novelty_long_scale",
                "multiscale_consensus",
                "novelty",
            }.issubset(set(result["feature_table"].columns))
        )
        self.assertIn("auto_labels", result["event_table"].columns)
        self.assertTrue(
            {
                "evidence_score",
                "evidence_label",
                "evidence_summary",
                "boundary_score",
                "multiscale_consensus",
            }.issubset(set(result["event_table"].columns))
        )
        self.assertTrue(result["event_table"]["evidence_score"].between(0.0, 1.0).all())
        self.assertEqual(result["analysis_metadata"]["method_version"], "soundscape-method-2.2")
        self.assertEqual(len(result["analysis_metadata"]["source_sha256"]), 64)
        self.assertEqual(len(result["analysis_metadata"]["analysis_audio_sha256"]), 64)
        self.assertIn("section_fingerprint_table", result)
        self.assertIn("state_similarity_table", result)
        self.assertGreaterEqual(float(result["spectrogram_db"].min()), -90.0)
        self.assertLessEqual(float(result["spectrogram_db"].max()), 0.0)

    def test_event_and_section_summaries_reuse_whole_piece_frame_grid(self) -> None:
        sr = 8000
        time_axis = np.arange(sr * 4, dtype=np.float32) / sr
        audio = np.concatenate(
            [
                0.08 * np.sin(2 * np.pi * 180 * time_axis[: sr * 2]),
                0.16 * np.sin(2 * np.pi * 1400 * time_axis[: sr * 2]),
            ]
        ).astype(np.float32)

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "frame-reuse.wav"
            sf.write(path, audio, sr)
            with mock.patch(
                "spectral_tool.analysis._segment_features",
                side_effect=AssertionError("local STFT should not be recomputed"),
            ):
                result = analyze_audio(
                    path,
                    config=AnalysisConfig(
                        n_fft=1024,
                        hop_length=256,
                        threshold_sigma=0.4,
                        prominence_sigma=0.3,
                        min_event_distance_sec=0.5,
                    ),
                    source_sha256="a" * 64,
                )

        self.assertGreaterEqual(len(result["event_table"]), 1)
        self.assertGreaterEqual(len(result["section_table"]), 1)

    def test_precomputed_source_sha256_avoids_metadata_reread(self) -> None:
        sr = 8000
        audio = np.zeros(sr, dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "prehashed.wav"
            sf.write(path, audio, sr)
            with mock.patch(
                "spectral_tool.analysis._source_sha256",
                side_effect=AssertionError("source hash should be reused"),
            ):
                result = analyze_audio(
                    path,
                    config=AnalysisConfig(multiscale_enabled=False),
                    source_sha256="b" * 64,
                )

        self.assertEqual(result["analysis_metadata"]["source_sha256"], "b" * 64)

    def test_long_form_boundaries_are_not_capped_at_six_and_states_recur(self) -> None:
        rng = np.random.default_rng(11)
        sr = 22050
        segment_duration = 1.5
        sample_count = int(sr * segment_duration)
        time_axis = np.linspace(0.0, segment_duration, sample_count, endpoint=False)
        segments: list[np.ndarray] = []
        for index in range(9):
            if index % 3 == 0:
                segment = 0.13 * np.sin(2 * np.pi * 150 * time_axis)
            elif index % 3 == 1:
                segment = 0.08 * rng.normal(size=sample_count) + 0.04 * np.sin(2 * np.pi * 2600 * time_axis)
            else:
                segment = 0.14 * np.sin(2 * np.pi * 800 * time_axis) + 0.03 * rng.normal(size=sample_count)
            segments.append(segment.astype(np.float32))
        audio = np.concatenate(segments)

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "recurrent-states.wav"
            sf.write(path, audio, sr)
            result = analyze_audio(
                path,
                config=AnalysisConfig(
                    n_fft=2048,
                    hop_length=512,
                    smooth_sigma=0.8,
                    threshold_sigma=0.35,
                    prominence_sigma=0.25,
                    min_event_distance_sec=0.8,
                    context_window_sec=0.6,
                    event_model_preset="timbre_soundscape",
                ),
            )

        self.assertGreaterEqual(int(result["event_table"]["is_major_boundary"].sum()), 7)
        self.assertGreaterEqual(len(result["section_fingerprint_table"]), 8)
        self.assertTrue(result["state_similarity_table"]["is_recurrence_candidate"].any())
        recurrence = result["state_similarity_table"].loc[
            result["state_similarity_table"]["is_recurrence_candidate"]
        ]
        self.assertGreaterEqual(float(recurrence["similarity_score"].max()), 0.78)

    def test_multiscale_detection_can_be_disabled_without_changing_output_schema(self) -> None:
        sr = 22050
        time_axis = np.linspace(0.0, 3.0, sr * 3, endpoint=False)
        audio = np.concatenate(
            [
                0.12 * np.sin(2 * np.pi * 180 * time_axis[: sr]),
                0.08 * np.sin(2 * np.pi * 1400 * time_axis[: sr]),
                0.04 * np.sin(2 * np.pi * 180 * time_axis[: sr]),
            ]
        ).astype(np.float32)

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "single-scale.wav"
            sf.write(path, audio, sr)
            result = analyze_audio(
                path,
                config=AnalysisConfig(
                    n_fft=1024,
                    hop_length=256,
                    min_event_distance_sec=0.5,
                    multiscale_enabled=False,
                ),
            )

        self.assertTrue(np.allclose(result["feature_table"]["multiscale_consensus"], 1.0))
        self.assertIn("novelty_short_scale", result["feature_table"].columns)

    def test_very_short_audio_uses_a_safe_stft_window(self) -> None:
        sr = 8000
        audio = np.sin(2 * np.pi * 440 * np.arange(96) / sr).astype(np.float32)

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "very-short.wav"
            sf.write(path, audio, sr)
            result = analyze_audio(
                path,
                config=AnalysisConfig(n_fft=4096, hop_length=1024),
            )

        self.assertGreater(len(result["feature_table"]), 0)
        self.assertLessEqual(int(result["hop_length"]), 24)
        self.assertEqual(result["analysis_metadata"]["source_name"], "very-short.wav")

    def test_mp3_falls_back_to_audioread(self) -> None:
        class NamedBytesIO(io.BytesIO):
            name = "demo.mp3"

        class FakeAudioHandle:
            channels = 2
            samplerate = 44100

            def __enter__(self) -> "FakeAudioHandle":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def __iter__(self):
                pcm = np.array([0, 1000, -1000, 0, 500, -500], dtype="<i2")
                yield pcm.tobytes()

        source = NamedBytesIO(b"fake-mp3-data")

        with mock.patch("spectral_tool.analysis.sf.read", side_effect=RuntimeError("unsupported")):
            with mock.patch("spectral_tool.analysis.audioread.audio_open", return_value=FakeAudioHandle()):
                audio, sr = _load_audio(source, target_sr=None)

        self.assertEqual(sr, 44100)
        self.assertEqual(audio.shape[0], 2)
        self.assertEqual(audio.shape[1], 3)
        self.assertTrue(np.all(np.abs(audio) <= 1.0))

    def test_m4a_falls_back_to_audioread(self) -> None:
        class NamedBytesIO(io.BytesIO):
            name = "demo.m4a"

        class FakeAudioHandle:
            channels = 2
            samplerate = 48000

            def __enter__(self) -> "FakeAudioHandle":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def __iter__(self):
                pcm = np.array([0, 1200, -1200, 0, 600, -600], dtype="<i2")
                yield pcm.tobytes()

        source = NamedBytesIO(b"fake-m4a-data")

        with mock.patch("spectral_tool.analysis.sf.read", side_effect=RuntimeError("unsupported")):
            with mock.patch("spectral_tool.analysis.audioread.audio_open", return_value=FakeAudioHandle()):
                audio, sr = _load_audio(source, target_sr=None)

        self.assertEqual(sr, 48000)
        self.assertEqual(audio.shape[0], 2)
        self.assertEqual(audio.shape[1], 3)
        self.assertTrue(np.all(np.abs(audio) <= 1.0))

    def test_uploaded_flac_remains_open_and_uses_requested_analysis_rate(self) -> None:
        class NamedBytesIO(io.BytesIO):
            name = "uploaded-recording.flac"

        source_sr = 44100
        duration_sec = 2.0
        time_axis = np.arange(int(source_sr * duration_sec), dtype=np.float32) / source_sr
        stereo = np.column_stack(
            [
                0.10 * np.sin(2 * np.pi * 220 * time_axis),
                0.08 * np.sin(2 * np.pi * 440 * time_axis),
            ]
        ).astype(np.float32)
        source = NamedBytesIO()
        sf.write(source, stereo, source_sr, format="FLAC", subtype="PCM_24")
        source.seek(0)

        result = analyze_audio(
            source,
            config=AnalysisConfig(target_sr=32000, multiscale_enabled=False),
            channel_mode="mix",
        )

        self.assertFalse(source.closed)
        self.assertEqual(result["sr"], 32000)
        self.assertAlmostEqual(result["duration_sec"], duration_sec, places=3)
        self.assertEqual(result["analysis_metadata"]["source_channel_count"], 2)
        self.assertEqual(result["analysis_metadata"]["source_name"], source.name)
        self.assertEqual(result["audio"].shape[0], 1)

    def test_resolve_novelty_weights_uses_custom_preset_mix(self) -> None:
        config = AnalysisConfig(
            novelty_weight_cosine=0.45,
            novelty_weight_flux=0.30,
            novelty_weight_onset=0.15,
            novelty_weight_rms=0.10,
            event_model_preset="timbre_soundscape",
        )

        weights = _resolve_novelty_weights(config)

        self.assertEqual(weights, (0.45, 0.30, 0.15, 0.10))

    def test_timbre_soundscape_mode_uses_extra_timbre_features(self) -> None:
        sr = 22050
        segment_duration = 2.0
        sample_count = int(sr * segment_duration)
        time_axis = np.linspace(0.0, segment_duration, sample_count, endpoint=False)
        rng = np.random.default_rng(19)

        low_tone = 0.14 * np.sin(2 * np.pi * 180 * time_axis)
        bright_noise = 0.05 * rng.normal(size=sample_count) + 0.08 * np.sin(2 * np.pi * 3400 * time_axis)
        audio = np.concatenate([low_tone, bright_noise]).astype(np.float32)

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "timbre-mode.wav"
            sf.write(path, audio, sr)

            common_kwargs = dict(
                target_sr=22050,
                n_fft=2048,
                hop_length=512,
                smooth_sigma=1.0,
                threshold_sigma=0.8,
                prominence_sigma=0.5,
                min_event_distance_sec=0.8,
                context_window_sec=1.0,
                novelty_weight_cosine=0.45,
                novelty_weight_flux=0.30,
                novelty_weight_onset=0.15,
                novelty_weight_rms=0.10,
            )

            balanced = analyze_audio(
                path,
                config=AnalysisConfig(
                    event_model_preset="balanced_default",
                    **common_kwargs,
                ),
            )
            timbre = analyze_audio(
                path,
                config=AnalysisConfig(
                    event_model_preset="timbre_soundscape",
                    **common_kwargs,
                ),
            )

        self.assertIn("band_energy_ratio", timbre["feature_table"].columns)
        self.assertIn("high_band_ratio", timbre["feature_table"].columns)
        self.assertFalse(np.allclose(balanced["novelty"], timbre["novelty"]))

    def test_build_audio_excerpt_wav_returns_playable_bytes(self) -> None:
        sr = 22050
        time_axis = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
        left = 0.2 * np.sin(2 * np.pi * 220 * time_axis)
        right = 0.2 * np.sin(2 * np.pi * 440 * time_axis)
        stereo = np.vstack([left, right]).astype(np.float32)

        clip = build_audio_excerpt_wav(stereo, sr, 0.5, 1.0, channel_mode="mix")
        self.assertGreater(len(clip), 100)

        decoded, decoded_sr = sf.read(io.BytesIO(clip), always_2d=True, dtype="float32")
        self.assertEqual(decoded_sr, sr)
        self.assertGreater(decoded.shape[0], 0)

    def test_plot_local_spectrogram_returns_figure(self) -> None:
        spectrogram_db = np.random.default_rng(1).normal(size=(32, 60)).astype(np.float32)
        times = np.linspace(0.0, 12.0, 60, dtype=np.float32)
        freqs = np.linspace(0.0, 12000.0, 32, dtype=np.float32)

        figure = plot_local_spectrogram(
            spectrogram_db=spectrogram_db,
            times=times,
            freqs=freqs,
            center_time=6.0,
            window_radius_sec=2.0,
            highlight_time=6.0,
            max_freq_hz=8000.0,
        )

        self.assertGreaterEqual(len(figure.axes), 1)

    def test_plot_local_spectrogram_can_recompute_from_audio_excerpt(self) -> None:
        sr = 22050
        time_axis = np.arange(sr * 4, dtype=np.float32) / sr
        samples = (
            0.15 * np.sin(2 * np.pi * 440 * time_axis)
            + 0.08 * np.sin(2 * np.pi * 1800 * time_axis)
        ).astype(np.float32)

        figure = plot_local_spectrogram(
            spectrogram_db=np.empty((0, 0), dtype=np.float32),
            times=np.array([], dtype=np.float32),
            freqs=np.array([], dtype=np.float32),
            center_time=2.0,
            window_radius_sec=1.0,
            samples=samples,
            sr=sr,
            n_fft=2048,
            hop_length=512,
        )

        self.assertIn("局部峰值 = 0 dB", figure.axes[0].get_title())
        self.assertGreater(len(figure.axes[0].collections), 0)

    def test_plot_local_waveform_returns_figure(self) -> None:
        sr = 22050
        time_axis = np.linspace(0.0, 6.0, sr * 6, endpoint=False)
        samples = (0.15 * np.sin(2 * np.pi * 220 * time_axis)).astype(np.float32)

        figure = plot_local_waveform(
            samples=samples,
            sr=sr,
            center_time=3.0,
            window_radius_sec=1.5,
            highlight_time=3.0,
        )

        self.assertGreaterEqual(len(figure.axes), 1)

    def test_build_local_waveform_chart_returns_plotly_figure(self) -> None:
        sr = 22050
        time_axis = np.linspace(0.0, 4.0, sr * 4, endpoint=False)
        samples = (0.12 * np.sin(2 * np.pi * 330 * time_axis)).astype(np.float32)

        figure = build_local_waveform_chart(
            samples=samples,
            sr=sr,
            center_time=2.0,
            window_radius_sec=1.0,
            highlight_time=2.0,
        )

        self.assertGreaterEqual(len(figure.data), 1)

    def test_build_synced_waveform_player_html_contains_audio_and_playhead_logic(self) -> None:
        sr = 22050
        time_axis = np.linspace(0.0, 3.0, sr * 3, endpoint=False)
        samples = (0.08 * np.sin(2 * np.pi * 220 * time_axis)).astype(np.float32)
        audio_bytes = build_audio_excerpt_wav(samples, sr, 0.0, 2.0, channel_mode="mix")

        html = build_synced_waveform_player_html(
            audio_bytes=audio_bytes,
            samples=samples,
            sr=sr,
            clip_start_sec=0.5,
            clip_end_sec=2.5,
            event_time_sec=1.5,
        )

        self.assertIn("data:audio/wav;base64,", html)
        self.assertIn("audio.currentTime", html)
        self.assertIn("点击波形可跳转", html)

    def test_build_synced_overview_player_html_contains_dual_canvas_and_controls(self) -> None:
        sr = 22050
        time_axis = np.linspace(0.0, 3.0, sr * 3, endpoint=False)
        samples = (0.08 * np.sin(2 * np.pi * 220 * time_axis)).astype(np.float32)
        audio_bytes = build_audio_excerpt_wav(samples, sr, 0.0, 2.0, channel_mode="mix")
        spectrogram_db = np.random.default_rng(3).normal(size=(48, 120)).astype(np.float32)
        spec_times = np.linspace(0.0, 2.0, 120, dtype=np.float32)
        spec_freqs = np.linspace(0.0, 26000.0, 48, dtype=np.float32)

        html = build_synced_overview_player_html(
            audio_bytes=audio_bytes,
            audio_mime="audio/wav",
            samples=samples,
            sr=sr,
            spectrogram_db=spectrogram_db,
            spectrogram_times=spec_times,
            spectrogram_freqs=spec_freqs,
            duration_sec=3.0,
        )

        self.assertIn("overviewWaveCanvas", html)
        self.assertIn("overviewSpecCanvas", html)
        self.assertIn("频率上限", html)
        self.assertIn("聚焦窗口", html)
        self.assertIn("overviewViewModeToggle", html)
        self.assertIn("切换到局部聚焦", html)
        self.assertIn("当前聚焦区间", html)
        self.assertIn("overviewWaveControl", html)
        self.assertIn("overviewFreqControl", html)
        self.assertIn("slider-vertical", html)
        self.assertIn("overviewWaveAmplitudeMax", html)
        self.assertIn("波形纵轴上限", html)
        self.assertIn('max="26000"', html)
        self.assertIn("频率 (Hz)", html)
        self.assertIn("振幅", html)
        self.assertIn("悬停波形", html)
        self.assertIn("包络振幅", html)
        self.assertIn("悬停频谱", html)
        self.assertIn("updateWaveHover", html)
        self.assertIn("updateSpecHover", html)
        self.assertIn("zoomAroundPointer", html)
        self.assertIn("toggleOverviewMode", html)
        self.assertIn("pointerdown", html)
        self.assertIn("dblclick", html)

        log_html = build_synced_overview_player_html(
            audio_bytes=audio_bytes,
            audio_mime="audio/wav",
            samples=samples,
            sr=sr,
            spectrogram_db=spectrogram_db,
            spectrogram_times=spec_times,
            spectrogram_freqs=spec_freqs,
            duration_sec=3.0,
            frequency_scale="log",
        )
        self.assertIn('const frequencyScale = "log"', log_html)
        self.assertIn("频率 (Hz，对数)", log_html)

    def test_build_event_assistant_payload_for_major_boundary(self) -> None:
        payload = build_event_assistant_payload(
            {
                "time_label": "03:24.10",
                "strength": 1.26,
                "prominence": 0.74,
                "pre_rms": 0.12,
                "post_rms": 0.23,
                "post_centroid_hz": 3620.0,
                "post_rolloff_hz": 8610.0,
                "post_high_ratio": 0.34,
                "post_flatness": 0.27,
                "channel_bias": "左侧偏强",
                "dominant_freqs": "312Hz, 1288Hz, 3810Hz",
                "is_major_boundary": True,
                "interaction_priority": "strong",
                "auto_labels": f"高频扩展 | {CANDIDATE_BOUNDARY_LABEL}",
            },
            effective_label_text=f"高频扩展 | {CANDIDATE_BOUNDARY_LABEL}",
        )

        self.assertIn("像不像边界", payload["question"])
        self.assertEqual(payload["role_text"], CANDIDATE_BOUNDARY_LABEL)
        self.assertIn("候选", payload["draft"])
        self.assertEqual(len(payload["primary_actions"]), 3)
        self.assertIn("看它是不是边界", [item["label"] for item in payload["primary_actions"]])
        self.assertIn("看这个变化有没有持续下去", payload["suggested_checks"])
        self.assertEqual(payload["interaction_priority"], "strong")
        self.assertEqual(payload["interaction_depth"], "full")

    def test_build_event_assistant_payload_includes_time_and_labels(self) -> None:
        payload = build_event_assistant_payload(
            {
                "time_label": "01:07.30",
                "strength": 0.85,
                "prominence": 0.44,
                "pre_rms": 0.08,
                "post_rms": 0.11,
                "post_centroid_hz": 2140.0,
                "post_rolloff_hz": 5520.0,
                "post_high_ratio": 0.18,
                "post_flatness": 0.11,
                "channel_bias": "平衡",
                "dominant_freqs": "440Hz, 1760Hz",
                "is_major_boundary": False,
                "interaction_priority": "weak",
                "auto_labels": "新事件出现 | 音色突变",
            },
            effective_label_text="新事件出现 | 音色突变",
        )

        self.assertIn("01:07.30", payload["draft"])
        self.assertIn("新事件出现", payload["label_text"])
        self.assertGreaterEqual(len(payload["evidence_points"]), 4)
        self.assertEqual(payload["assistant_invite"], "AI 小助手")
        self.assertEqual(payload["assistant_invite_hint"], "先看一眼就好")
        self.assertIn("先给我一个轻量观察", [item["label"] for item in payload["primary_actions"]])
        self.assertIn("AI 设置", payload["settings_title"])
        self.assertEqual(payload["interaction_depth"], "light")

    def test_annotate_event_interaction_levels_adds_three_priority_levels(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "strength": 0.22,
                    "prominence": 0.08,
                    "is_major_boundary": False,
                    "auto_labels": "新事件出现",
                    "effective_labels": "新事件出现",
                },
                {
                    "event_id": 2,
                    "strength": 0.61,
                    "prominence": 0.34,
                    "is_major_boundary": False,
                    "auto_labels": "新事件出现 | 音色突变",
                    "effective_labels": "新事件出现 | 音色突变",
                },
                {
                    "event_id": 3,
                    "strength": 1.31,
                    "prominence": 0.86,
                    "is_major_boundary": True,
                    "auto_labels": f"高频扩展 | {CANDIDATE_BOUNDARY_LABEL}",
                    "effective_labels": f"高频扩展 | {CANDIDATE_BOUNDARY_LABEL}",
                },
            ]
        )

        annotated = annotate_event_interaction_levels(frame)

        self.assertIn("interaction_priority", annotated.columns)
        self.assertIn("interaction_priority_label", annotated.columns)
        self.assertIn("interaction_depth", annotated.columns)
        self.assertIn("weak", set(annotated["interaction_priority"]))
        self.assertIn("medium", set(annotated["interaction_priority"]))
        self.assertIn("strong", set(annotated["interaction_priority"]))
        self.assertEqual(
            annotated.loc[annotated["event_id"] == 3, "interaction_priority"].iloc[0],
            "strong",
        )

    def test_annotate_event_similarity_groups_clusters_highly_similar_events(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "post_rms": 0.121,
                    "post_centroid_hz": 2210.0,
                    "post_bandwidth_hz": 1460.0,
                    "post_rolloff_hz": 5280.0,
                    "post_low_ratio": 0.20,
                    "post_mid_ratio": 0.49,
                    "post_high_ratio": 0.31,
                    "post_flatness": 0.14,
                },
                {
                    "event_id": 2,
                    "post_rms": 0.122,
                    "post_centroid_hz": 2225.0,
                    "post_bandwidth_hz": 1455.0,
                    "post_rolloff_hz": 5300.0,
                    "post_low_ratio": 0.201,
                    "post_mid_ratio": 0.488,
                    "post_high_ratio": 0.311,
                    "post_flatness": 0.141,
                },
                {
                    "event_id": 3,
                    "post_rms": 0.042,
                    "post_centroid_hz": 620.0,
                    "post_bandwidth_hz": 380.0,
                    "post_rolloff_hz": 1300.0,
                    "post_low_ratio": 0.72,
                    "post_mid_ratio": 0.23,
                    "post_high_ratio": 0.05,
                    "post_flatness": 0.03,
                },
            ]
        )

        annotated = annotate_event_similarity_groups(frame, threshold=0.90)

        group_1 = int(annotated.loc[annotated["event_id"] == 1, "similarity_group_id"].iloc[0])
        group_2 = int(annotated.loc[annotated["event_id"] == 2, "similarity_group_id"].iloc[0])
        group_3 = int(annotated.loc[annotated["event_id"] == 3, "similarity_group_id"].iloc[0])

        self.assertEqual(group_1, group_2)
        self.assertNotEqual(group_1, group_3)
        self.assertEqual(int(annotated.loc[annotated["event_id"] == 1, "similarity_group_size"].iloc[0]), 2)
        self.assertGreaterEqual(float(annotated.loc[annotated["event_id"] == 1, "max_similarity_in_group"].iloc[0]), 0.90)

    def test_identical_event_states_have_full_similarity(self) -> None:
        row = {
            "post_rms": 0.12,
            "post_centroid_hz": 1800.0,
            "post_bandwidth_hz": 900.0,
            "post_rolloff_hz": 4200.0,
            "post_low_ratio": 0.25,
            "post_mid_ratio": 0.55,
            "post_high_ratio": 0.20,
            "post_flatness": 0.10,
        }
        frame = pd.DataFrame([{"event_id": 1, **row}, {"event_id": 2, **row}])

        annotated = annotate_event_similarity_groups(frame, threshold=0.99)

        self.assertEqual(annotated["similarity_group_id"].nunique(), 1)
        self.assertTrue(np.allclose(annotated["max_similarity_in_group"], 1.0))

    def test_build_assistant_overlay_component_targets_parent_document(self) -> None:
        payload = build_event_assistant_payload(
            {
                "time_label": "03:24.10",
                "strength": 1.26,
                "prominence": 0.74,
                "pre_rms": 0.12,
                "post_rms": 0.23,
                "post_centroid_hz": 3620.0,
                "post_rolloff_hz": 8610.0,
                "post_high_ratio": 0.34,
                "post_flatness": 0.27,
                "channel_bias": "左侧偏强",
                "dominant_freqs": "312Hz, 1288Hz, 3810Hz",
                "is_major_boundary": True,
                "auto_labels": f"高频扩展 | {CANDIDATE_BOUNDARY_LABEL}",
                "descriptor": f"高频扩展；{CANDIDATE_BOUNDARY_LABEL}",
            },
            effective_label_text=f"高频扩展 | {CANDIDATE_BOUNDARY_LABEL}",
        )

        html = build_assistant_overlay_component(3, payload)

        self.assertIn("window.parent.document", html)
        self.assertIn("event-ai-overlay-root", html)
        self.assertIn("AI 小助手", html)
        self.assertIn("建议重点讨论", html)
        self.assertIn("先给我一个初步观察", html)
        self.assertIn("看前后有没有变", html)
        self.assertIn("看它是不是边界", html)
        self.assertIn("AI 设置", html)
        self.assertIn("展开更详细的分析", html)
        self.assertIn("Collaborative Analysis", html)
        self.assertIn("重置位置", html)
        self.assertIn("拖动可移动", html)
        self.assertIn(CANDIDATE_BOUNDARY_LABEL, html)

    def test_boundary_state_change_score_rejects_pure_local_spike(self) -> None:
        before = {
            "rms": 0.12,
            "centroid_hz": 1800.0,
            "bandwidth_hz": 1200.0,
            "rolloff_hz": 5200.0,
            "flatness": 0.10,
            "low_ratio": 0.22,
            "mid_ratio": 0.53,
            "high_ratio": 0.25,
        }
        after = {
            "rms": 0.17,
            "centroid_hz": 1830.0,
            "bandwidth_hz": 1215.0,
            "rolloff_hz": 5260.0,
            "flatness": 0.11,
            "low_ratio": 0.21,
            "mid_ratio": 0.52,
            "high_ratio": 0.27,
        }

        score, is_clear_change = _boundary_state_change_score(before, after)

        self.assertLess(score, 1.05)
        self.assertFalse(is_clear_change)

    def test_boundary_state_change_score_accepts_clear_state_shift(self) -> None:
        before = {
            "rms": 0.09,
            "centroid_hz": 920.0,
            "bandwidth_hz": 640.0,
            "rolloff_hz": 2100.0,
            "flatness": 0.05,
            "low_ratio": 0.62,
            "mid_ratio": 0.30,
            "high_ratio": 0.08,
        }
        after = {
            "rms": 0.18,
            "centroid_hz": 2700.0,
            "bandwidth_hz": 1560.0,
            "rolloff_hz": 6900.0,
            "flatness": 0.18,
            "low_ratio": 0.21,
            "mid_ratio": 0.43,
            "high_ratio": 0.36,
        }

        score, is_clear_change = _boundary_state_change_score(before, after)

        self.assertGreaterEqual(score, 1.05)
        self.assertTrue(is_clear_change)


if __name__ == "__main__":
    unittest.main()
