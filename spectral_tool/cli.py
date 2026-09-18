from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import AnalysisConfig, analyze_audio
from .visualization import (
    build_summary_text,
    plot_event_density,
    plot_novelty,
    plot_spectrogram,
    plot_waveform,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="声景/频谱事件自动标记工具")
    parser.add_argument("input", type=Path, help="输入音频文件路径，支持 WAV/FLAC/AIFF/MP3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports"),
        help="结果导出目录，默认为 exports/",
    )
    parser.add_argument(
        "--channel",
        choices=["mix", "left", "right"],
        default="mix",
        help="分析通道，默认混合分析",
    )
    parser.add_argument("--target-sr", type=int, default=22050, help="目标采样率，默认 22050")
    parser.add_argument("--n-fft", type=int, default=4096, help="频谱窗长，默认 4096")
    parser.add_argument("--hop-length", type=int, default=1024, help="帧移，默认 1024")
    parser.add_argument("--smooth-sigma", type=float, default=1.6, help="新颖度平滑强度")
    parser.add_argument("--threshold-sigma", type=float, default=1.0, help="检测阈值强度")
    parser.add_argument("--prominence-sigma", type=float, default=0.8, help="峰值显著度强度")
    parser.add_argument("--min-event-distance", type=float, default=5.0, help="最小事件间隔（秒）")
    parser.add_argument("--context-window", type=float, default=3.0, help="事件前后比较窗口（秒）")
    parser.add_argument(
        "--disable-multiscale",
        action="store_true",
        help="关闭短、中、长时间尺度共识检测",
    )
    parser.add_argument(
        "--multiscale-weight",
        type=float,
        default=0.35,
        help="多尺度证据在新颖度曲线中的权重，默认 0.35",
    )
    parser.add_argument(
        "--spectrogram-dynamic-range",
        type=float,
        default=90.0,
        help="频谱图固定动态范围（dB），默认 90",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AnalysisConfig(
        target_sr=args.target_sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        smooth_sigma=args.smooth_sigma,
        threshold_sigma=args.threshold_sigma,
        prominence_sigma=args.prominence_sigma,
        min_event_distance_sec=args.min_event_distance,
        context_window_sec=args.context_window,
        multiscale_enabled=not args.disable_multiscale,
        multiscale_weight=args.multiscale_weight,
        spectrogram_dynamic_range_db=args.spectrogram_dynamic_range,
    )

    result = analyze_audio(args.input, config=config, channel_mode=args.channel)
    stem_dir = args.output_dir / args.input.stem
    stem_dir.mkdir(parents=True, exist_ok=True)

    result["event_table"].to_csv(stem_dir / "events.csv", index=False, encoding="utf-8-sig")
    result["section_table"].to_csv(stem_dir / "sections.csv", index=False, encoding="utf-8-sig")
    result["feature_table"].to_csv(stem_dir / "feature_table.csv", index=False, encoding="utf-8-sig")
    result["section_fingerprint_table"].to_csv(
        stem_dir / "soundscape_state_fingerprints.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result["state_similarity_table"].to_csv(
        stem_dir / "soundscape_state_similarities.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "config": result["config"],
        "analysis_metadata": result["analysis_metadata"],
        "sr": result["sr"],
        "duration_sec": result["duration_sec"],
        "channel_mode": result["channel_mode"],
        "summary_lines": result["summary_lines"],
        "events": result["event_table"].to_dict(orient="records"),
        "sections": result["section_table"].to_dict(orient="records"),
        "section_fingerprints": result["section_fingerprint_table"].to_dict(orient="records"),
        "state_similarities": result["state_similarity_table"].to_dict(orient="records"),
        "feature_table": result["feature_table"].to_dict(orient="records"),
    }
    (stem_dir / "analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (stem_dir / "summary.txt").write_text(
        build_summary_text(result["summary_lines"]),
        encoding="utf-8",
    )

    save_figure(
        plot_waveform(result["selected_audio"], result["sr"], result["peak_times"]),
        stem_dir / "waveform_events.png",
    )
    save_figure(
        plot_novelty(
            result["times"],
            result["novelty"],
            result["threshold"],
            result["peak_indices"],
        ),
        stem_dir / "novelty_curve.png",
    )
    save_figure(
        plot_spectrogram(
            result["spectrogram_db"],
            result["spectrogram_times"],
            result["spectrogram_freqs"],
            result["peak_times"],
        ),
        stem_dir / "spectrogram.png",
    )
    save_figure(
        plot_event_density(result["peak_times"], result["duration_sec"]),
        stem_dir / "event_density.png",
    )

    print(f"分析完成，结果已导出到: {stem_dir}")
    print(f"检测到事件数量: {len(result['event_table'])}")
    print(build_summary_text(result["summary_lines"]))


if __name__ == "__main__":
    main()
