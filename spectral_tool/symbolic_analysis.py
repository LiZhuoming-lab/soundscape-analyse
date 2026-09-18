from __future__ import annotations

from bisect import bisect_right
from itertools import combinations
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
from music21 import chord as m21chord
from music21 import converter
from music21 import interval as m21interval
from music21 import key as m21key
from music21 import meter as m21meter
from music21 import note as m21note
from music21 import pitch as m21pitch
from music21 import roman as m21roman
from music21 import stream as m21stream


@dataclass(slots=True)
class SymbolicAnalysisConfig:
    theme_window_notes: int = 6
    max_recurrence_results: int = 16
    measure_summary_top_n: int = 3

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


KEYBOARD_PART_KEYWORDS = ("piano", "klavier", "pf", "keyboard", "harpsichord", "organ", "钢琴")

VERTICAL_INTERVAL_CONSONANCE_RANKS = {
    "纯一度": 1,
    "纯八度": 2,
    "纯五度": 3,
    "纯四度": 4,
    "大三度": 5,
    "小三度": 6,
    "大六度": 7,
    "小六度": 8,
    "大二度": 9,
    "小七度": 10,
    "小二度": 11,
    "大七度": 12,
    "增四度 / 减五度": 13,
}


def _format_pitch_name_chinese(name: str) -> str:
    token = str(name or "").strip()
    if not token:
        return ""
    if token.endswith("--"):
        return f"重降{token[:-2]}"
    if token.endswith("##"):
        return f"重升{token[:-2]}"
    if token.endswith("-"):
        return f"降{token[:-1]}"
    if token.endswith("#"):
        return f"升{token[:-1]}"
    return token


def format_key_label(key_object: m21key.Key | None) -> str:
    if key_object is None:
        return "未能稳定判断"

    mode_map = {
        "major": "大调",
        "minor": "小调",
        "dorian": "多利亚调式",
        "phrygian": "弗里几亚调式",
        "lydian": "利底亚调式",
        "mixolydian": "混合利底亚调式",
        "aeolian": "爱奥利亚调式",
        "locrian": "洛克里亚调式",
    }
    mode_label = mode_map.get(str(key_object.mode), str(key_object.mode))
    return f"{_format_pitch_name_chinese(key_object.tonic.name)}{mode_label}"


def analyze_symbolic_score(
    source: str | Path | BinaryIO | m21stream.Score,
    config: SymbolicAnalysisConfig | None = None,
) -> dict[str, Any]:
    active_config = config or SymbolicAnalysisConfig()
    score, source_name = _load_symbolic_score(source)
    parts = list(score.parts) or [score]
    global_key = _detect_global_key(score)

    note_table, melodic_sequences, part_summary = _build_note_table(parts, global_key)
    pitch_class_histogram = _build_pitch_class_histogram(note_table)
    pitch_height_histogram = _build_pitch_height_histogram(note_table)
    scale_degree_histogram = _build_scale_degree_histogram(note_table)
    measure_pitch_summary = _build_measure_pitch_summary(note_table, active_config.measure_summary_top_n)
    interval_table, interval_class_histogram, directed_interval_histogram = _build_interval_tables(melodic_sequences)
    harmony_table = _build_harmony_table(score, global_key)
    vertical_interval_table = _build_vertical_interval_table(harmony_table)
    vertical_interval_histogram = _build_vertical_interval_histogram(vertical_interval_table)
    measure_consonance_summary = _build_measure_consonance_summary(vertical_interval_table)
    theme_sequences = _select_theme_search_sequences(melodic_sequences)
    cadence_candidates = _build_cadence_candidates(
        harmony_table=harmony_table,
        note_table=note_table,
        melodic_sequences=theme_sequences,
        global_key=global_key,
    )
    theme_matches = _build_theme_matches(
        melodic_sequences=theme_sequences,
        window_size=active_config.theme_window_notes,
        max_results=active_config.max_recurrence_results,
    )

    total_notes = int(len(note_table))
    unique_pitches = int(note_table["midi"].nunique()) if not note_table.empty else 0
    unique_pitch_classes = int(note_table["pitch_class"].nunique()) if not note_table.empty else 0

    summary_lines = _build_summary_lines(
        global_key=global_key,
        total_notes=total_notes,
        unique_pitches=unique_pitches,
        unique_pitch_classes=unique_pitch_classes,
        pitch_class_histogram=pitch_class_histogram,
        interval_class_histogram=interval_class_histogram,
        vertical_interval_histogram=vertical_interval_histogram,
        harmony_table=harmony_table,
        cadence_candidates=cadence_candidates,
        theme_matches=theme_matches,
        theme_window_notes=active_config.theme_window_notes,
    )

    return {
        "source_name": source_name,
        "score_title": _resolve_score_title(score, source_name),
        "config": active_config.to_dict(),
        "global_key": format_key_label(global_key),
        "global_key_object": global_key,
        "part_summary": part_summary,
        "note_table": note_table,
        "pitch_class_histogram": pitch_class_histogram,
        "pitch_height_histogram": pitch_height_histogram,
        "scale_degree_histogram": scale_degree_histogram,
        "measure_pitch_summary": measure_pitch_summary,
        "interval_table": interval_table,
        "interval_class_histogram": interval_class_histogram,
        "directed_interval_histogram": directed_interval_histogram,
        "vertical_interval_table": vertical_interval_table,
        "vertical_interval_histogram": vertical_interval_histogram,
        "measure_consonance_summary": measure_consonance_summary,
        "harmony_table": harmony_table,
        "cadence_candidates": cadence_candidates,
        "theme_matches": theme_matches,
        "summary_lines": summary_lines,
        "total_notes": total_notes,
        "unique_pitches": unique_pitches,
        "unique_pitch_classes": unique_pitch_classes,
    }


def _resolve_source_name(source: str | Path | BinaryIO | m21stream.Score) -> str:
    if isinstance(source, Path):
        return source.name
    if isinstance(source, str):
        return Path(source).name
    if isinstance(source, m21stream.Score):
        metadata_title = getattr(getattr(source, "metadata", None), "title", None)
        return str(metadata_title or "score_input")
    name = getattr(source, "name", None)
    if isinstance(name, str) and name:
        return Path(name).name
    return "score_input"


def _write_temp_symbolic_file(source: BinaryIO) -> str:
    source.seek(0)
    data = source.read()
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("乐谱输入必须是路径、music21 Score 或字节流。")

    suffix = Path(_resolve_source_name(source)).suffix or ".xml"
    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temporary.write(data)
        temporary.flush()
    finally:
        temporary.close()
    return temporary.name


def _load_symbolic_score(
    source: str | Path | BinaryIO | m21stream.Score,
) -> tuple[m21stream.Score, str]:
    source_name = _resolve_source_name(source)
    temp_path: str | None = None

    try:
        if isinstance(source, m21stream.Score):
            parsed = source
        elif isinstance(source, (str, Path)):
            parsed = converter.parse(str(source))
        else:
            temp_path = _write_temp_symbolic_file(source)
            parsed = converter.parse(temp_path)
    except Exception as exc:
        raise RuntimeError(
            "无法读取该乐谱文件。请尝试 MusicXML、MXL、MIDI 或标准 XML 乐谱文件。"
        ) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    if isinstance(parsed, m21stream.Score):
        score = parsed
    else:
        score = m21stream.Score(id="ImportedScore")
        score.insert(0, parsed)

    return score, source_name


def _detect_global_key(score: m21stream.Score) -> m21key.Key | None:
    explicit_keys = list(score.recurse().getElementsByClass(m21key.Key))
    if explicit_keys:
        return explicit_keys[0]

    try:
        analyzed = score.analyze("key")
        if isinstance(analyzed, m21key.Key):
            return analyzed
    except Exception:
        return None
    return None


def _resolve_score_title(score: m21stream.Score, source_name: str) -> str:
    metadata = getattr(score, "metadata", None)
    if metadata is not None:
        title = getattr(metadata, "title", None)
        movement_name = getattr(metadata, "movementName", None)
        if title:
            return str(title)
        if movement_name:
            return str(movement_name)
    return source_name


def _pitch_from_midi(midi_value: int) -> m21pitch.Pitch:
    pitch_object = m21pitch.Pitch()
    pitch_object.midi = int(midi_value)
    return pitch_object


def _pitch_class_name(pitch_class: int) -> str:
    pitch_object = _pitch_from_midi(60 + int(pitch_class))
    return pitch_object.name


def _base_part_name(part_name: str) -> str:
    return re.sub(r"\s*\[\d+\]$", "", str(part_name)).strip()


def _is_keyboard_part_name(part_name: str) -> bool:
    lowered = _base_part_name(part_name).lower()
    return any(keyword in lowered for keyword in KEYBOARD_PART_KEYWORDS)


def _select_theme_search_sequences(
    melodic_sequences: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    if len(melodic_sequences) <= 1:
        return melodic_sequences

    grouped_names: dict[str, list[str]] = {}
    for part_name in melodic_sequences:
        grouped_names.setdefault(_base_part_name(part_name), []).append(part_name)

    selected: dict[str, list[dict[str, Any]]] = {}
    for base_name, part_names in grouped_names.items():
        if len(part_names) > 1 and _is_keyboard_part_name(base_name):
            selected[part_names[0]] = melodic_sequences[part_names[0]]
            continue
        for part_name in part_names:
            selected[part_name] = melodic_sequences[part_name]
    return selected


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_scale_degree(key_object: m21key.Key | None, pitch_object: m21pitch.Pitch) -> int | None:
    if key_object is None:
        return None
    candidates: list[m21pitch.Pitch] = [pitch_object]
    try:
        enharmonic = pitch_object.getEnharmonic()
        if enharmonic is not None:
            candidates.append(enharmonic)
    except Exception:
        pass
    for candidate in candidates:
        try:
            degree = key_object.getScaleDegreeFromPitch(candidate)
        except Exception:
            degree = None
        if degree is not None:
            return degree
    return None


def _build_reference_measure_timeline(parts: list[m21stream.Stream]) -> list[dict[str, float | int]]:
    reference_part: m21stream.Stream | None = None
    reference_measure_count = -1
    for part in parts:
        measures = list(part.getElementsByClass("Measure"))
        if len(measures) > reference_measure_count:
            reference_part = part
            reference_measure_count = len(measures)

    if reference_part is None:
        return []

    timeline: list[dict[str, float | int]] = []
    for measure in reference_part.getElementsByClass("Measure"):
        start_offset = _safe_float(getattr(measure, "offset", 0.0), 0.0)
        measure_duration = _safe_float(getattr(measure, "quarterLength", 0.0), 0.0)
        if measure_duration <= 0.0:
            continue
        time_signature = measure.getContextByClass(m21meter.TimeSignature)
        beat_length = _safe_float(
            getattr(getattr(time_signature, "beatDuration", None), "quarterLength", 1.0),
            1.0,
        )
        timeline.append(
            {
                "measure_number": _safe_int(getattr(measure, "number", 0), 0),
                "start_offset": start_offset,
                "end_offset": start_offset + measure_duration,
                "beat_length": beat_length if beat_length > 0.0 else 1.0,
            }
        )
    return timeline


def _map_offset_to_measure(
    absolute_offset: float,
    timeline: list[dict[str, float | int]],
) -> tuple[int, float]:
    if not timeline:
        return 0, 1.0

    starts = [float(item["start_offset"]) for item in timeline]
    index = max(0, bisect_right(starts, float(absolute_offset) + 1e-9) - 1)
    index = min(index, len(timeline) - 1)
    item = timeline[index]

    start_offset = float(item["start_offset"])
    end_offset = float(item["end_offset"])
    beat_length = float(item["beat_length"]) or 1.0

    if float(absolute_offset) >= end_offset - 1e-9 and index + 1 < len(timeline):
        item = timeline[index + 1]
        start_offset = float(item["start_offset"])
        beat_length = float(item["beat_length"]) or 1.0

    local_offset = max(0.0, float(absolute_offset) - start_offset)
    beat = round(local_offset / beat_length + 1.0, 3)
    return int(item["measure_number"]), beat


def _build_note_table(
    parts: list[m21stream.Stream],
    global_key: m21key.Key | None,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]], pd.DataFrame]:
    note_rows: list[dict[str, Any]] = []
    melodic_sequences: dict[str, list[dict[str, Any]]] = {}
    part_rows: list[dict[str, Any]] = []
    part_name_counts: dict[str, int] = {}
    reference_timeline = _build_reference_measure_timeline(parts)

    for part_index, part in enumerate(parts, start=1):
        base_part_name = str(part.partName or part.id or f"Part {part_index}")
        part_name_counts[base_part_name] = part_name_counts.get(base_part_name, 0) + 1
        if part_name_counts[base_part_name] == 1:
            part_name = base_part_name
        else:
            part_name = f"{base_part_name} [{part_name_counts[base_part_name]}]"
        flattened = part.flatten().notes
        onset_index = 0
        melodic_events: list[dict[str, Any]] = []

        for element in flattened:
            onset_index += 1
            offset_ql = round(_safe_float(getattr(element, "offset", 0.0), 0.0), 3)
            mapped_measure_number, mapped_beat = _map_offset_to_measure(offset_ql, reference_timeline)
            raw_measure_number = _safe_int(getattr(element, "measureNumber", 0), 0)
            raw_beat = round(_safe_float(getattr(element, "beat", 0.0), 0.0), 3)
            measure_number = mapped_measure_number or raw_measure_number
            beat = mapped_beat if mapped_measure_number > 0 else raw_beat
            quarter_length = round(_safe_float(getattr(element, "quarterLength", 0.0), 0.0), 3)

            if isinstance(element, m21chord.Chord):
                event_pitches = [pitch_value for pitch_value in element.pitches if getattr(pitch_value, "midi", None) is not None]
                if not event_pitches:
                    continue
                melodic_pitch = max(event_pitches, key=lambda pitch_value: pitch_value.midi)
                is_chord_event = True
            elif isinstance(element, m21note.Note):
                if getattr(element.pitch, "midi", None) is None:
                    continue
                event_pitches = [element.pitch]
                melodic_pitch = element.pitch
                is_chord_event = False
            else:
                continue

            melodic_events.append(
                {
                    "part_name": part_name,
                    "onset_index": onset_index,
                    "measure_number": measure_number,
                    "beat": beat,
                    "offset_ql": offset_ql,
                    "quarter_length": quarter_length,
                    "midi": int(melodic_pitch.midi),
                    "pitch_name": melodic_pitch.nameWithOctave,
                    "pitch_class": int(melodic_pitch.pitchClass),
                    "scale_degree": _safe_scale_degree(global_key, melodic_pitch),
                }
            )

            chord_cardinality = len(event_pitches)
            for chord_tone_index, pitch_object in enumerate(event_pitches, start=1):
                note_rows.append(
                    {
                        "part_name": part_name,
                        "onset_index": onset_index,
                        "measure_number": measure_number,
                        "beat": beat,
                        "offset_ql": offset_ql,
                        "quarter_length": quarter_length,
                        "pitch_name": pitch_object.nameWithOctave,
                        "pitch_class_label": _pitch_class_name(int(pitch_object.pitchClass)),
                        "midi": int(pitch_object.midi),
                        "pitch_class": int(pitch_object.pitchClass),
                        "octave": _safe_int(pitch_object.octave, 0),
                        "scale_degree": _safe_scale_degree(global_key, pitch_object),
                        "scale_degree_label": (
                            str(_safe_scale_degree(global_key, pitch_object))
                            if _safe_scale_degree(global_key, pitch_object) is not None
                            else ""
                        ),
                        "is_chord_event": is_chord_event,
                        "chord_cardinality": chord_cardinality,
                        "chord_tone_index": chord_tone_index,
                    }
                )

        melodic_sequences[part_name] = melodic_events
        part_rows.append(
            {
                "part_name": part_name,
                "event_count": len(melodic_events),
                "note_count": len([row for row in note_rows if row["part_name"] == part_name]),
            }
        )

    note_table = pd.DataFrame(note_rows)
    part_summary = pd.DataFrame(part_rows)
    if not note_table.empty:
        note_table = note_table.sort_values(["measure_number", "offset_ql", "part_name", "midi"]).reset_index(drop=True)
    return note_table, melodic_sequences, part_summary


def _build_pitch_class_histogram(note_table: pd.DataFrame) -> pd.DataFrame:
    if note_table.empty:
        return pd.DataFrame(columns=["pitch_class", "pitch_class_label", "count", "ratio"])

    histogram = (
        note_table.groupby(["pitch_class", "pitch_class_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["pitch_class"])
        .reset_index(drop=True)
    )
    histogram["ratio"] = histogram["count"] / float(histogram["count"].sum())
    return histogram


def _build_pitch_height_histogram(note_table: pd.DataFrame) -> pd.DataFrame:
    if note_table.empty:
        return pd.DataFrame(columns=["midi", "pitch_name", "count"])

    histogram = (
        note_table.groupby(["midi", "pitch_name"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["midi", "pitch_name"])
        .reset_index(drop=True)
    )
    return histogram


def _build_scale_degree_histogram(note_table: pd.DataFrame) -> pd.DataFrame:
    if note_table.empty or "scale_degree" not in note_table.columns:
        return pd.DataFrame(columns=["scale_degree", "count", "ratio"])

    filtered = note_table.dropna(subset=["scale_degree"]).copy()
    if filtered.empty:
        return pd.DataFrame(columns=["scale_degree", "count", "ratio"])

    filtered["scale_degree"] = filtered["scale_degree"].astype(int)
    histogram = (
        filtered.groupby("scale_degree", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("scale_degree")
        .reset_index(drop=True)
    )
    histogram["ratio"] = histogram["count"] / float(histogram["count"].sum())
    return histogram


def _top_counts(series: pd.Series, top_n: int) -> str:
    valid = series.astype(str)
    valid = valid[valid != ""]
    if valid.empty:
        return ""
    counts = valid.value_counts().head(top_n)
    return " / ".join(f"{label}×{count}" for label, count in counts.items())


def _build_measure_pitch_summary(note_table: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if note_table.empty:
        return pd.DataFrame(
            columns=[
                "measure_number",
                "note_count",
                "unique_pitch_count",
                "unique_pitch_classes",
                "lowest_pitch",
                "highest_pitch",
                "top_pitch_classes",
                "top_scale_degrees",
            ]
        )

    rows: list[dict[str, Any]] = []
    for measure_number, frame in note_table.groupby("measure_number", sort=True):
        low_midi = int(frame["midi"].min())
        high_midi = int(frame["midi"].max())
        rows.append(
            {
                "measure_number": int(measure_number),
                "note_count": int(len(frame)),
                "unique_pitch_count": int(frame["midi"].nunique()),
                "unique_pitch_classes": int(frame["pitch_class"].nunique()),
                "lowest_pitch": _pitch_from_midi(low_midi).nameWithOctave,
                "highest_pitch": _pitch_from_midi(high_midi).nameWithOctave,
                "top_pitch_classes": _top_counts(frame["pitch_class_label"], top_n),
                "top_scale_degrees": _top_counts(frame["scale_degree_label"], top_n),
            }
        )
    return pd.DataFrame(rows)


def _interval_class(semitones: int) -> int:
    absolute = abs(int(semitones)) % 12
    return min(absolute, 12 - absolute)


def _build_interval_tables(
    melodic_sequences: dict[str, list[dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for part_name, events in melodic_sequences.items():
        if len(events) < 2:
            continue
        for previous, current in zip(events[:-1], events[1:]):
            previous_pitch = _pitch_from_midi(int(previous["midi"]))
            current_pitch = _pitch_from_midi(int(current["midi"]))
            interval_object = m21interval.Interval(previous_pitch, current_pitch)
            semitones = int(current["midi"]) - int(previous["midi"])
            contour = "上行" if semitones > 0 else "下行" if semitones < 0 else "同音重复"
            rows.append(
                {
                    "part_name": part_name,
                    "from_measure": int(previous["measure_number"]),
                    "to_measure": int(current["measure_number"]),
                    "from_pitch": str(previous["pitch_name"]),
                    "to_pitch": str(current["pitch_name"]),
                    "semitones": semitones,
                    "interval_class": _interval_class(semitones),
                    "simple_name": interval_object.simpleName,
                    "directed_name": interval_object.directedName,
                    "contour": contour,
                }
            )

    interval_table = pd.DataFrame(rows)
    if interval_table.empty:
        empty_ic = pd.DataFrame(columns=["interval_class", "count", "ratio"])
        empty_dir = pd.DataFrame(columns=["directed_name", "count", "ratio"])
        return interval_table, empty_ic, empty_dir

    interval_class_histogram = (
        interval_table.groupby("interval_class", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("interval_class")
        .reset_index(drop=True)
    )
    interval_class_histogram["ratio"] = interval_class_histogram["count"] / float(interval_class_histogram["count"].sum())

    directed_interval_histogram = (
        interval_table.groupby("directed_name", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "directed_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    directed_interval_histogram["ratio"] = directed_interval_histogram["count"] / float(
        directed_interval_histogram["count"].sum()
    )
    return interval_table, interval_class_histogram, directed_interval_histogram


def _safe_root_pitch(chord_object: m21chord.Chord) -> m21pitch.Pitch | None:
    try:
        return chord_object.root()
    except Exception:
        return None


def _build_harmony_table(
    score: m21stream.Score,
    global_key: m21key.Key | None,
) -> pd.DataFrame:
    try:
        harmonic_stream = score.chordify()
    except Exception:
        return pd.DataFrame(
            columns=[
                "slice_id",
                "measure_number",
                "beat",
                "offset_ql",
                "quarter_length",
                "pitch_names",
                "pitch_class_set",
                "root",
                "root_scale_degree",
                "bass",
                "bass_pitch_name",
                "bass_scale_degree",
                "quality",
                "roman_numeral",
            ]
        )

    rows: list[dict[str, Any]] = []
    slice_id = 0
    for chord_object in harmonic_stream.flatten().getElementsByClass(m21chord.Chord):
        if len(chord_object.pitches) == 0:
            continue

        slice_id += 1
        unique_pitch_classes = sorted({int(pitch_value.pitchClass) for pitch_value in chord_object.pitches})
        root_pitch = _safe_root_pitch(chord_object)
        roman_numeral = ""
        if global_key is not None and len(chord_object.pitches) >= 2:
            try:
                roman_numeral = str(m21roman.romanNumeralFromChord(chord_object, global_key).figure)
            except Exception:
                roman_numeral = ""

        rows.append(
            {
                "slice_id": slice_id,
                "measure_number": _safe_int(getattr(chord_object, "measureNumber", 0), 0),
                "beat": round(_safe_float(getattr(chord_object, "beat", 0.0), 0.0), 3),
                "offset_ql": round(_safe_float(getattr(chord_object, "offset", 0.0), 0.0), 3),
                "quarter_length": round(_safe_float(getattr(chord_object, "quarterLength", 0.0), 0.0), 3),
                "pitch_names": " ".join(pitch_value.nameWithOctave for pitch_value in chord_object.pitches),
                "pitch_class_set": "{" + ", ".join(_pitch_class_name(pitch_class) for pitch_class in unique_pitch_classes) + "}",
                "root": root_pitch.name if root_pitch is not None else "",
                "root_scale_degree": (
                    _safe_scale_degree(global_key, root_pitch) if root_pitch is not None else None
                ),
                "bass": chord_object.bass().name if chord_object.bass() is not None else "",
                "bass_pitch_name": chord_object.bass().nameWithOctave if chord_object.bass() is not None else "",
                "bass_scale_degree": (
                    _safe_scale_degree(global_key, chord_object.bass()) if chord_object.bass() is not None else None
                ),
                "quality": str(getattr(chord_object, "quality", "") or ""),
                "roman_numeral": roman_numeral,
            }
        )

    harmony_table = pd.DataFrame(rows)
    if harmony_table.empty:
        return harmony_table
    return harmony_table.sort_values(["measure_number", "beat", "slice_id"]).reset_index(drop=True)


def _classify_vertical_interval_consonance(interval_object: m21interval.Interval) -> tuple[str, int]:
    semitones = abs(int(interval_object.semitones))
    if semitones == 0:
        return "纯一度", VERTICAL_INTERVAL_CONSONANCE_RANKS["纯一度"]
    if semitones % 12 == 0:
        return "纯八度", VERTICAL_INTERVAL_CONSONANCE_RANKS["纯八度"]

    mapping = {
        "P5": "纯五度",
        "P4": "纯四度",
        "M3": "大三度",
        "m3": "小三度",
        "M6": "大六度",
        "m6": "小六度",
        "M2": "大二度",
        "m7": "小七度",
        "m2": "小二度",
        "M7": "大七度",
        "A4": "增四度 / 减五度",
        "d5": "增四度 / 减五度",
    }
    interval_label = mapping.get(interval_object.simpleName)
    if interval_label is None:
        fallback_map = {
            1: "小二度",
            2: "大二度",
            3: "小三度",
            4: "大三度",
            5: "纯四度",
            6: "增四度 / 减五度",
            7: "纯五度",
            8: "小六度",
            9: "大六度",
            10: "小七度",
            11: "大七度",
        }
        interval_label = fallback_map.get(semitones % 12, "增四度 / 减五度")
    return interval_label, int(VERTICAL_INTERVAL_CONSONANCE_RANKS[interval_label])


def _build_vertical_interval_table(harmony_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "slice_id",
        "measure_number",
        "beat",
        "quarter_length",
        "lower_pitch",
        "upper_pitch",
        "interval_name",
        "simple_name",
        "semitones",
        "consonance_rank",
        "consonance_label",
    ]
    if harmony_table.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for _, harmony_row in harmony_table.iterrows():
        tokens = [token for token in str(harmony_row.get("pitch_names", "")).split() if token]
        if len(tokens) < 2:
            continue
        pitches: list[m21pitch.Pitch] = []
        for token in tokens:
            try:
                pitches.append(m21pitch.Pitch(token))
            except Exception:
                continue
        if len(pitches) < 2:
            continue

        ordered_pitches = sorted(pitches, key=lambda pitch_object: int(pitch_object.midi))
        for lower_pitch, upper_pitch in combinations(ordered_pitches, 2):
            interval_object = m21interval.Interval(lower_pitch, upper_pitch)
            interval_label, consonance_rank = _classify_vertical_interval_consonance(interval_object)
            rows.append(
                {
                    "slice_id": int(harmony_row["slice_id"]),
                    "measure_number": int(harmony_row["measure_number"]),
                    "beat": round(_safe_float(harmony_row["beat"]), 3),
                    "quarter_length": round(_safe_float(harmony_row["quarter_length"]), 3),
                    "lower_pitch": lower_pitch.nameWithOctave,
                    "upper_pitch": upper_pitch.nameWithOctave,
                    "interval_name": interval_label,
                    "simple_name": interval_object.simpleName,
                    "semitones": abs(int(interval_object.semitones)),
                    "consonance_rank": consonance_rank,
                    "consonance_label": f"{consonance_rank} = {interval_label}",
                }
            )

    vertical_interval_table = pd.DataFrame(rows)
    if vertical_interval_table.empty:
        return pd.DataFrame(columns=columns)
    return vertical_interval_table.sort_values(
        ["measure_number", "beat", "slice_id", "consonance_rank", "lower_pitch", "upper_pitch"]
    ).reset_index(drop=True)


def _build_vertical_interval_histogram(vertical_interval_table: pd.DataFrame) -> pd.DataFrame:
    if vertical_interval_table.empty:
        return pd.DataFrame(columns=["interval_name", "consonance_rank", "count", "ratio"])

    histogram = (
        vertical_interval_table.groupby(["interval_name", "consonance_rank"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["consonance_rank", "interval_name"])
        .reset_index(drop=True)
    )
    histogram["ratio"] = histogram["count"] / float(histogram["count"].sum())
    return histogram


def _build_measure_consonance_summary(vertical_interval_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "measure_number",
        "vertical_interval_count",
        "mean_consonance_rank",
        "lowest_consonance_rank",
        "highest_consonance_rank",
        "consonant_interval_count",
        "dissonant_interval_count",
        "top_vertical_intervals",
    ]
    if vertical_interval_table.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for measure_number, frame in vertical_interval_table.groupby("measure_number", sort=True):
        ranks = frame["consonance_rank"].astype(int)
        rows.append(
            {
                "measure_number": int(measure_number),
                "vertical_interval_count": int(len(frame)),
                "mean_consonance_rank": round(float(ranks.mean()), 3),
                "lowest_consonance_rank": int(ranks.min()),
                "highest_consonance_rank": int(ranks.max()),
                "consonant_interval_count": int((ranks <= 8).sum()),
                "dissonant_interval_count": int((ranks >= 9).sum()),
                "top_vertical_intervals": _top_counts(frame["interval_name"], 3),
            }
        )
    return pd.DataFrame(rows)


def _roman_is_dominant(roman_figure: str) -> bool:
    return _roman_primary_degree(roman_figure) in {"V", "v"}


def _roman_is_tonic(roman_figure: str) -> bool:
    return _roman_primary_degree(roman_figure) in {"I", "i"}


def _is_strong_beat(beat: float) -> bool:
    return _is_close_to_value(beat, 1.0) or _is_close_to_value(beat, 3.0)


def _select_primary_melodic_events(
    melodic_sequences: dict[str, list[dict[str, Any]]],
) -> tuple[str | None, list[dict[str, Any]]]:
    if not melodic_sequences:
        return None, []
    first_part_name = next(iter(melodic_sequences.keys()))
    return first_part_name, melodic_sequences[first_part_name]


def _measure_lowest_pitch_map(note_table: pd.DataFrame) -> dict[int, dict[str, Any]]:
    return _measure_window_lowest_pitch_map(note_table)


def _measure_window_lowest_pitch_map(
    note_table: pd.DataFrame,
    min_beat: float | None = None,
    max_beat: float | None = None,
) -> dict[int, dict[str, Any]]:
    if note_table.empty:
        return {}

    filtered = note_table.copy()
    if min_beat is not None:
        filtered = filtered.loc[filtered["beat"].astype(float) >= float(min_beat) - 1e-6]
    if max_beat is not None:
        filtered = filtered.loc[filtered["beat"].astype(float) <= float(max_beat) + 1e-6]
    if filtered.empty:
        return {}

    ordered = filtered.sort_values(["measure_number", "midi", "offset_ql", "part_name"]).reset_index(drop=True)
    lowest_rows = ordered.groupby("measure_number", as_index=False).first()
    measure_map: dict[int, dict[str, Any]] = {}
    for _, row in lowest_rows.iterrows():
        measure_map[int(row["measure_number"])] = {
            "pitch_name": str(row["pitch_name"]),
            "midi": int(row["midi"]),
            "part_name": str(row["part_name"]),
        }
    return measure_map


def _dedupe_key_candidates(candidates: list[m21key.Key]) -> list[m21key.Key]:
    deduped: list[m21key.Key] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        token = (candidate.tonic.name, str(candidate.mode))
        if token in seen:
            continue
        seen.add(token)
        deduped.append(candidate)
    return deduped


def _candidate_cadence_keys(
    current_measure: int,
    measure_rows_map: dict[int, pd.DataFrame],
    primary_events_by_measure: dict[int, list[dict[str, Any]]],
    opening_lowest_map: dict[int, dict[str, Any]],
    global_key: m21key.Key | None,
) -> list[m21key.Key]:
    current_rows = _slice_measure_frame(measure_rows_map.get(current_measure), max_beat=2.0)
    next_rows = _slice_measure_frame(measure_rows_map.get(current_measure + 1), max_beat=1.5)

    tonic_name_candidates: list[str] = []
    for frame in [current_rows, next_rows]:
        for column_name in ["root", "bass_pitch_name"]:
            if frame.empty or column_name not in frame.columns:
                continue
            for value in frame[column_name].astype(str):
                pitch_name = _normalize_pitch_name(value)
                if pitch_name:
                    tonic_name_candidates.append(pitch_name)

    opening_lowest = opening_lowest_map.get(current_measure)
    if opening_lowest is not None:
        tonic_name_candidates.append(_normalize_pitch_name(str(opening_lowest["pitch_name"])))

    for measure_number in [current_measure, current_measure + 1]:
        for event in primary_events_by_measure.get(measure_number, []):
            beat = _safe_float(event.get("beat"), 0.0)
            duration = _safe_float(event.get("quarter_length"), 0.0)
            if beat <= 2.0 or duration >= 1.0:
                tonic_name_candidates.append(_normalize_pitch_name(str(event.get("pitch_name", ""))))

    arrival_pitch_classes = set()
    arrival_pitch_classes.update(_frame_pitch_classes(current_rows))
    arrival_pitch_classes.update(_frame_pitch_classes(next_rows))
    if global_key is not None:
        arrival_pitch_classes.add(int(global_key.tonic.pitchClass))

    candidates: list[m21key.Key] = []
    for tonic_name in tonic_name_candidates:
        if not tonic_name:
            continue
        for mode in _infer_candidate_modes_for_tonic(tonic_name, arrival_pitch_classes, global_key):
            try:
                candidates.append(m21key.Key(tonic_name, mode))
            except Exception:
                continue
    if global_key is not None:
        candidates.append(global_key)
    return _dedupe_key_candidates(candidates)


def _safe_chord_from_pitch_names(pitch_names: str) -> m21chord.Chord | None:
    tokens = [token for token in str(pitch_names).split() if token]
    if not tokens:
        return None
    try:
        return m21chord.Chord(tokens)
    except Exception:
        return None


def _safe_roman_figure_for_key(chord_object: m21chord.Chord | None, key_object: m21key.Key | None) -> str:
    if chord_object is None or key_object is None or len(chord_object.pitches) < 2:
        return ""
    try:
        return str(m21roman.romanNumeralFromChord(chord_object, key_object).figure)
    except Exception:
        return ""


def _active_melodic_event_at_offset(
    melodic_events: list[dict[str, Any]],
    offset_ql: float,
    slice_duration: float,
) -> dict[str, Any] | None:
    slice_end = float(offset_ql) + max(float(slice_duration), 1e-6)
    overlapping: list[dict[str, Any]] = []
    for event in melodic_events:
        event_start = float(event["offset_ql"])
        event_end = event_start + max(float(event["quarter_length"]), 1e-6)
        if event_start <= float(offset_ql) + 1e-6 and event_end > float(offset_ql) + 1e-6:
            overlapping.append(event)
        elif float(offset_ql) - 1e-6 <= event_start < slice_end - 1e-6:
            overlapping.append(event)
    if not overlapping:
        return None
    return max(overlapping, key=lambda item: (float(item["offset_ql"]), int(item["midi"])))


def _slice_measure_frame(
    frame: pd.DataFrame | None,
    min_beat: float | None = None,
    max_beat: float | None = None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=frame.columns if frame is not None else [])
    sliced = frame.copy()
    if min_beat is not None:
        sliced = sliced.loc[sliced["beat"].astype(float) >= float(min_beat) - 1e-6]
    if max_beat is not None:
        sliced = sliced.loc[sliced["beat"].astype(float) <= float(max_beat) + 1e-6]
    return sliced.reset_index(drop=True)


def _group_rows_by_measure(frame: pd.DataFrame) -> dict[int, pd.DataFrame]:
    if frame.empty:
        return {}
    return {
        int(measure_number): group.sort_values(["beat", "slice_id"]).reset_index(drop=True)
        for measure_number, group in frame.groupby("measure_number", sort=True)
    }


def _group_melodic_events_by_measure(events: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(int(event["measure_number"]), []).append(event)
    return grouped


def _normalize_pitch_name(value: str) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    try:
        return m21pitch.Pitch(token).name
    except Exception:
        return ""


def _frame_pitch_classes(frame: pd.DataFrame) -> set[int]:
    if frame is None or frame.empty or "pitch_names" not in frame.columns:
        return set()
    pitch_classes: set[int] = set()
    for pitch_names in frame["pitch_names"].astype(str):
        for token in pitch_names.split():
            try:
                pitch_classes.add(int(m21pitch.Pitch(token).pitchClass))
            except Exception:
                continue
    return pitch_classes


def _roman_pitch_class_set(roman_figure: str, key_object: m21key.Key | None) -> set[int]:
    if key_object is None:
        return set()
    try:
        rn = m21roman.RomanNumeral(roman_figure, key_object)
        return {int(pitch_value.pitchClass) for pitch_value in rn.pitches}
    except Exception:
        return set()


def _pitch_class_coverage(actual: set[int], expected: set[int]) -> float:
    if not actual or not expected:
        return 0.0
    return len(actual & expected) / float(len(expected))


def _infer_candidate_modes_for_tonic(
    tonic_name: str,
    pitch_classes: set[int],
    global_key: m21key.Key | None,
) -> list[str]:
    normalized = _normalize_pitch_name(tonic_name)
    if not normalized:
        return []

    tonic_pc = int(m21pitch.Pitch(normalized).pitchClass)
    major_third_pc = (tonic_pc + 4) % 12
    minor_third_pc = (tonic_pc + 3) % 12
    fifth_pc = (tonic_pc + 7) % 12

    candidates: list[str] = []
    if major_third_pc in pitch_classes and fifth_pc in pitch_classes:
        candidates.append("major")
    if minor_third_pc in pitch_classes and fifth_pc in pitch_classes:
        candidates.append("minor")
    if not candidates and major_third_pc in pitch_classes:
        candidates.append("major")
    if not candidates and minor_third_pc in pitch_classes:
        candidates.append("minor")
    if not candidates and global_key is not None and global_key.tonic.name == normalized:
        candidates.append(str(global_key.mode))
    if not candidates and fifth_pc in pitch_classes:
        candidates.extend(["major", "minor"])
    if not candidates:
        candidates.append("major")

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _roman_is_leading_tone(roman_figure: str) -> bool:
    return _roman_primary_degree(roman_figure).lower() == "vii"


def _roman_primary_degree(roman_figure: str) -> str:
    value = str(roman_figure or "").strip()
    if not value:
        return ""
    match = re.match(r"^[b#]*([IViv]+)", value)
    if match is None:
        return ""
    return match.group(1)


def _roman_is_literal_tonic_arrival(roman_figure: str) -> bool:
    value = str(roman_figure or "").strip()
    return value in {"I", "i"}


def _display_tonic_label_for_key(key_object: m21key.Key) -> str:
    return "I（和声还原）" if str(key_object.mode) == "major" else "i（和声还原）"


def _score_tonic_arrival(
    current_measure: int,
    cadence_key: m21key.Key,
    measure_rows_map: dict[int, pd.DataFrame],
) -> dict[str, Any]:
    current_rows = _slice_measure_frame(measure_rows_map.get(current_measure), max_beat=2.0)
    next_rows = _slice_measure_frame(measure_rows_map.get(current_measure + 1), max_beat=1.5)
    tonic_pc_set = _roman_pitch_class_set("I" if str(cadence_key.mode) == "major" else "i", cadence_key)

    best_score = 0.0
    best_row: pd.Series | None = None
    best_rn = ""
    best_is_restored = False

    for frame in [current_rows, next_rows]:
        if frame.empty:
            continue
        for _, row in frame.iterrows():
            row_pitch_classes = _frame_pitch_classes(pd.DataFrame([row]))
            coverage = _pitch_class_coverage(row_pitch_classes, tonic_pc_set)
            row_chord = _safe_chord_from_pitch_names(row.get("pitch_names", ""))
            row_rn = _safe_roman_figure_for_key(row_chord, cadence_key)
            row_root = _normalize_pitch_name(str(row.get("root", "")))
            row_bass = _normalize_pitch_name(str(row.get("bass_pitch_name", "")))

            row_score = 0.0
            row_is_restored = False
            if _roman_is_tonic(row_rn):
                row_score = 0.44 if len(row_pitch_classes) == 1 else 0.76
            elif coverage >= (2.0 / 3.0):
                row_score = 0.36 + 0.30 * coverage
                row_is_restored = True
            elif coverage >= (1.0 / 3.0) and (row_root == cadence_key.tonic.name or row_bass == cadence_key.tonic.name):
                row_score = 0.28 + 0.22 * coverage
                row_is_restored = True

            if row_score <= 0.0:
                continue

            if _safe_int(row.get("measure_number"), 0) == current_measure:
                row_score += 0.12
            else:
                row_score += 0.04
            if _safe_float(row.get("beat"), 0.0) <= 1.5:
                row_score += 0.08
            if _safe_float(row.get("quarter_length"), 0.0) >= 1.0:
                row_score += 0.08

            if row_score > best_score:
                best_score = row_score
                best_row = row
                best_rn = row_rn
                best_is_restored = row_is_restored

    combined_pitch_classes = _frame_pitch_classes(current_rows) | _frame_pitch_classes(next_rows)
    combined_coverage = _pitch_class_coverage(combined_pitch_classes, tonic_pc_set)
    if combined_coverage >= 1.0:
        best_score += 0.20
    elif combined_coverage >= (2.0 / 3.0):
        best_score += 0.12
    elif combined_coverage >= (1.0 / 3.0):
        best_score += 0.05

    if best_row is None or best_score <= 0.0:
        return {
            "score": 0.0,
            "roman_numeral": "",
            "measure_number": current_measure,
            "beat": 1.0,
            "quarter_length": 0.0,
        }

    if best_is_restored and combined_coverage < (2.0 / 3.0):
        return {
            "score": 0.0,
            "roman_numeral": "",
            "measure_number": current_measure,
            "beat": 1.0,
            "quarter_length": 0.0,
        }

    display_rn = best_rn if _roman_is_tonic(best_rn) else _display_tonic_label_for_key(cadence_key)

    return {
        "score": min(best_score, 1.25),
        "roman_numeral": display_rn,
        "measure_number": _safe_int(best_row.get("measure_number"), current_measure),
        "beat": round(_safe_float(best_row.get("beat"), 1.0), 3),
        "quarter_length": round(_safe_float(best_row.get("quarter_length"), 0.0), 3),
    }


def _score_dominant_preparation(
    current_measure: int,
    cadence_key: m21key.Key,
    measure_rows_map: dict[int, pd.DataFrame],
) -> dict[str, Any]:
    dominant_pc_sets = [
        _roman_pitch_class_set("V", cadence_key),
        _roman_pitch_class_set("V7", cadence_key),
    ]
    leading_tone_pc_sets = [
        _roman_pitch_class_set("viio", cadence_key),
        _roman_pitch_class_set("viio7", cadence_key),
    ]

    best_score = 0.0
    best_row: pd.Series | None = None
    best_rn = ""

    for measure_number in range(max(1, current_measure - 2), current_measure):
        frame = measure_rows_map.get(measure_number)
        if frame is None or frame.empty:
            continue
        for _, row in frame.iterrows():
            row_pitch_classes = _frame_pitch_classes(pd.DataFrame([row]))
            dominant_coverage = max((_pitch_class_coverage(row_pitch_classes, pcs) for pcs in dominant_pc_sets if pcs), default=0.0)
            leading_coverage = max(
                (_pitch_class_coverage(row_pitch_classes, pcs) for pcs in leading_tone_pc_sets if pcs),
                default=0.0,
            )
            row_chord = _safe_chord_from_pitch_names(row.get("pitch_names", ""))
            row_rn = _safe_roman_figure_for_key(row_chord, cadence_key)

            row_score = 0.0
            if _roman_is_dominant(row_rn):
                row_score = 0.40 if len(row_pitch_classes) == 1 else 0.76
            elif _roman_is_leading_tone(row_rn):
                row_score = 0.26 if len(row_pitch_classes) == 1 else 0.58
            elif dominant_coverage >= (2.0 / 3.0):
                row_score = 0.34 + 0.30 * dominant_coverage
            elif leading_coverage >= (2.0 / 3.0):
                row_score = 0.28 + 0.24 * leading_coverage

            if row_score <= 0.0:
                continue

            if measure_number == current_measure - 1:
                row_score += 0.12
            else:
                row_score += 0.04

            beat = _safe_float(row.get("beat"), 0.0)
            if beat >= 2.5:
                row_score += 0.10
            elif beat >= 2.0:
                row_score += 0.05
            if _safe_float(row.get("quarter_length"), 0.0) >= 1.0:
                row_score += 0.06
            elif _safe_float(row.get("quarter_length"), 0.0) >= 0.5:
                row_score += 0.03

            if row_score > best_score:
                best_score = row_score
                best_row = row
                best_rn = row_rn

    previous_rows = _slice_measure_frame(measure_rows_map.get(current_measure - 1), min_beat=2.0)
    previous_pitch_classes = _frame_pitch_classes(previous_rows)
    dominant_window_coverage = max(
        (_pitch_class_coverage(previous_pitch_classes, pcs) for pcs in dominant_pc_sets if pcs),
        default=0.0,
    )
    if dominant_window_coverage >= 1.0:
        best_score += 0.18
    elif dominant_window_coverage >= (2.0 / 3.0):
        best_score += 0.10

    if best_row is None or best_score <= 0.0:
        return {
            "score": 0.0,
            "roman_numeral": "",
            "measure_number": current_measure - 1,
            "beat": 0.0,
        }

    return {
        "score": min(best_score, 1.20),
        "roman_numeral": best_rn or "V（和声还原）",
        "measure_number": _safe_int(best_row.get("measure_number"), current_measure - 1),
        "beat": round(_safe_float(best_row.get("beat"), 0.0), 3),
    }


def _score_tonic_line_arrival(
    current_measure: int,
    cadence_key: m21key.Key,
    primary_events_by_measure: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    candidate_events = list(primary_events_by_measure.get(current_measure, []))
    candidate_events.extend(
        event
        for event in primary_events_by_measure.get(current_measure + 1, [])
        if _safe_float(event.get("beat"), 0.0) <= 1.5
    )
    if not candidate_events:
        return {
            "score": 0.0,
            "roman_numeral": "",
            "measure_number": current_measure,
            "beat": 1.0,
            "quarter_length": 0.0,
        }

    annotated: list[tuple[dict[str, Any], int | None]] = []
    for event in candidate_events:
        pitch_object = _pitch_from_midi(_safe_int(event.get("midi"), 0))
        annotated.append((event, _safe_scale_degree(cadence_key, pitch_object)))

    best_score = 0.0
    best_event: dict[str, Any] | None = None
    for index, (event, degree) in enumerate(annotated):
        if degree != 1:
            continue
        score = 0.34
        beat = _safe_float(event.get("beat"), 0.0)
        duration = _safe_float(event.get("quarter_length"), 0.0)
        if _safe_int(event.get("measure_number"), current_measure) == current_measure:
            score += 0.08
        if beat <= 1.0 + 1e-6:
            score += 0.18
        elif beat <= 2.0 + 1e-6:
            score += 0.10
        if duration >= 1.0:
            score += 0.18
        elif duration >= 0.5:
            score += 0.10

        trailing_degrees = [item_degree for _, item_degree in annotated[index : index + 5] if item_degree is not None]
        if any(item_degree in {3, 5} for item_degree in trailing_degrees[1:]):
            score += 0.16
        if len({item_degree for item_degree in trailing_degrees if item_degree in {1, 3, 5}}) >= 2:
            score += 0.10

        if score > best_score:
            best_score = score
            best_event = event

    if best_event is None:
        return {
            "score": 0.0,
            "roman_numeral": "",
            "measure_number": current_measure,
            "beat": 1.0,
            "quarter_length": 0.0,
        }

    return {
        "score": min(best_score, 0.95),
        "roman_numeral": _display_tonic_label_for_key(cadence_key),
        "measure_number": _safe_int(best_event.get("measure_number"), current_measure),
        "beat": round(_safe_float(best_event.get("beat"), 1.0), 3),
        "quarter_length": round(_safe_float(best_event.get("quarter_length"), 0.0), 3),
    }


def _score_melodic_preparation(
    arrival_measure: int,
    cadence_key: m21key.Key,
    primary_events_by_measure: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    candidate_events = list(primary_events_by_measure.get(max(1, arrival_measure - 1), []))
    candidate_events.extend(primary_events_by_measure.get(arrival_measure, []))
    if not candidate_events:
        return {
            "score": 0.0,
            "roman_numeral": "",
            "measure_number": arrival_measure - 1,
            "beat": 0.0,
            "previous_degree": None,
            "distinct_degrees": tuple(),
        }

    annotated: list[tuple[dict[str, Any], int | None]] = []
    for event in candidate_events:
        pitch_object = _pitch_from_midi(_safe_int(event.get("midi"), 0))
        annotated.append((event, _safe_scale_degree(cadence_key, pitch_object)))

    arrival_index: int | None = None
    for index, (event, degree) in enumerate(annotated):
        if _safe_int(event.get("measure_number"), 0) != arrival_measure:
            continue
        if degree == 1:
            arrival_index = index
            break
    if arrival_index is None:
        return {
            "score": 0.0,
            "roman_numeral": "",
            "measure_number": arrival_measure - 1,
            "beat": 0.0,
            "previous_degree": None,
            "distinct_degrees": tuple(),
        }

    leading_context = [degree for _, degree in annotated[max(0, arrival_index - 6) : arrival_index] if degree is not None]
    if not leading_context:
        return {
            "score": 0.0,
            "roman_numeral": "",
            "measure_number": arrival_measure - 1,
            "beat": 0.0,
            "previous_degree": None,
            "distinct_degrees": tuple(),
        }

    immediate_previous = leading_context[-1]
    score = 0.0
    if immediate_previous == 7:
        score += 0.48
    elif immediate_previous == 2:
        score += 0.34
    elif immediate_previous == 5:
        score += 0.22
    if 5 in leading_context:
        score += 0.16
    if 7 in leading_context:
        score += 0.18
    if len(set(leading_context + [1])) >= 4 and 7 in leading_context:
        score += 0.12

    anchor_event = annotated[arrival_index - 1][0]
    return {
        "score": min(score, 0.90),
        "roman_numeral": "V（旋律准备）" if score >= 0.45 else "",
        "measure_number": _safe_int(anchor_event.get("measure_number"), arrival_measure - 1),
        "beat": round(_safe_float(anchor_event.get("beat"), 0.0), 3),
        "previous_degree": immediate_previous,
        "distinct_degrees": tuple(sorted(set(leading_context))),
    }


def _score_bass_support(
    previous_measure: int,
    current_measure: int,
    cadence_key: m21key.Key,
    measure_lowest_map: dict[int, dict[str, Any]],
    opening_lowest_map: dict[int, dict[str, Any]],
    closing_lowest_map: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    previous_primary = closing_lowest_map.get(previous_measure)
    previous_fallback = measure_lowest_map.get(previous_measure)
    current_primary = opening_lowest_map.get(current_measure)
    current_fallback = measure_lowest_map.get(current_measure)

    previous_pitch_name = ""
    previous_degree = None
    previous_score = 0.0
    for info, penalty in [(previous_primary, 0.0), (previous_fallback, 0.05)]:
        if info is None:
            continue
        pitch_object = _pitch_from_midi(int(info["midi"]))
        degree = _safe_scale_degree(cadence_key, pitch_object)
        score = 0.0
        if degree == 5:
            score = 0.44 - penalty
        elif degree in {2, 7}:
            score = 0.18 - penalty
        if score > previous_score:
            previous_score = score
            previous_degree = degree
            previous_pitch_name = str(info["pitch_name"])

    current_pitch_name = ""
    current_degree = None
    current_score = 0.0
    for info, penalty in [(current_primary, 0.0), (current_fallback, 0.05)]:
        if info is None:
            continue
        pitch_object = _pitch_from_midi(int(info["midi"]))
        degree = _safe_scale_degree(cadence_key, pitch_object)
        score = 0.0
        if degree == 1:
            score = 0.44 - penalty
        elif degree in {3, 5}:
            score = 0.16 - penalty
        if score > current_score:
            current_score = score
            current_degree = degree
            current_pitch_name = str(info["pitch_name"])

    return {
        "score": min(previous_score + current_score, 0.95),
        "previous_pitch_name": previous_pitch_name,
        "previous_degree": previous_degree,
        "current_pitch_name": current_pitch_name,
        "current_degree": current_degree,
    }


def _score_melodic_skeleton(
    current_measure: int,
    cadence_key: m21key.Key,
    primary_events_by_measure: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    candidate_events = list(primary_events_by_measure.get(current_measure, []))
    candidate_events.extend(
        event
        for event in primary_events_by_measure.get(current_measure + 1, [])
        if _safe_float(event.get("beat"), 0.0) <= 1.5
    )

    best_score = 0.0
    best_event: dict[str, Any] | None = None
    best_degree: int | None = None
    for event in candidate_events:
        pitch_object = _pitch_from_midi(_safe_int(event.get("midi"), 0))
        degree = _safe_scale_degree(cadence_key, pitch_object)
        if degree not in {1, 3, 5}:
            continue

        score = 0.72 if degree == 1 else 0.34
        beat = _safe_float(event.get("beat"), 0.0)
        duration = _safe_float(event.get("quarter_length"), 0.0)
        if int(event.get("measure_number", 0)) == current_measure:
            score += 0.08
        else:
            score += 0.02
        if beat <= 1.5:
            score += 0.10
        elif _is_strong_beat(beat):
            score += 0.07
        elif beat <= 2.5:
            score += 0.04
        if duration >= 1.0:
            score += 0.10
        elif duration >= 0.5:
            score += 0.05

        if score > best_score:
            best_score = score
            best_event = event
            best_degree = degree

    if best_event is None:
        return {
            "score": 0.0,
            "event": None,
            "scale_degree": None,
        }

    return {
        "score": min(best_score, 1.0),
        "event": best_event,
        "scale_degree": best_degree,
    }


def _score_cadential_closure(
    current_measure: int,
    previous_measure: int,
    cadence_key: m21key.Key,
    measure_rows_map: dict[int, pd.DataFrame],
    melody_event: dict[str, Any] | None,
    tonic_support: dict[str, Any],
) -> dict[str, Any]:
    current_opening = _slice_measure_frame(measure_rows_map.get(current_measure), max_beat=2.0)
    previous_closing = _slice_measure_frame(measure_rows_map.get(previous_measure), min_beat=2.0)
    if previous_closing.empty:
        previous_closing = measure_rows_map.get(previous_measure, pd.DataFrame())

    score = 0.0
    tonic_measure = _safe_int(tonic_support.get("measure_number"), current_measure)
    tonic_beat = _safe_float(tonic_support.get("beat"), 1.0)
    if tonic_measure == current_measure and tonic_beat <= 1.5:
        score += 0.18
    if melody_event is not None:
        melody_beat = _safe_float(melody_event.get("beat"), 0.0)
        melody_duration = _safe_float(melody_event.get("quarter_length"), 0.0)
        if melody_beat <= 1.5:
            score += 0.12
        if melody_duration >= 1.0:
            score += 0.18
        elif melody_duration >= 0.5:
            score += 0.08
    if previous_measure == current_measure - 1:
        score += 0.10
    if len(previous_closing) >= max(1, len(current_opening)):
        score += 0.10
    if not current_opening.empty and current_opening["quarter_length"].astype(float).max() >= 1.0:
        score += 0.10

    tonic_window = _slice_measure_frame(measure_rows_map.get(current_measure), max_beat=1.833)
    tonic_pc_set = _roman_pitch_class_set("I" if str(cadence_key.mode) == "major" else "i", cadence_key)
    tonic_like_count = 0
    harmonic_row_count = 0
    has_downbeat_tonic = False
    for _, row in tonic_window.iterrows():
        row_chord = _safe_chord_from_pitch_names(row.get("pitch_names", ""))
        if row_chord is None or len(row_chord.pitches) < 2:
            continue
        harmonic_row_count += 1
        row_rn = _safe_roman_figure_for_key(row_chord, cadence_key)
        row_pitch_classes = _frame_pitch_classes(pd.DataFrame([row]))
        row_coverage = _pitch_class_coverage(row_pitch_classes, tonic_pc_set)
        row_root = _normalize_pitch_name(str(row.get("root", "")))
        row_bass = _normalize_pitch_name(str(row.get("bass_pitch_name", "")))
        is_tonic_like = _roman_is_tonic(row_rn) or (
            row_coverage >= (2.0 / 3.0)
            and (row_root == cadence_key.tonic.name or row_bass == cadence_key.tonic.name)
        )
        if not is_tonic_like:
            continue
        tonic_like_count += 1
        if _safe_float(row.get("beat"), 0.0) <= 1.05:
            has_downbeat_tonic = True

    tonic_window_coverage = _pitch_class_coverage(_frame_pitch_classes(tonic_window), tonic_pc_set)
    if has_downbeat_tonic:
        score += 0.08
    if harmonic_row_count >= 3 and tonic_like_count >= 3 and tonic_window_coverage >= 1.0:
        score += 0.12
    elif tonic_like_count >= 2 and tonic_window_coverage >= 1.0:
        score += 0.06

    if score >= 0.55:
        strength_label = "高分候选"
    elif score >= 0.35:
        strength_label = "中高候选"
    else:
        strength_label = "中等候选"

    return {
        "score": min(score, 0.80),
        "strength_label": strength_label,
    }


def _melody_skeleton_class_label(scale_degree: int | None) -> str:
    if scale_degree == 1:
        return "主音骨架（1）"
    if scale_degree in {3, 5}:
        return f"非主音骨架（{scale_degree}）"
    return "骨架未定"


def _build_perfect_authentic_cadence_candidates(
    harmony_table: pd.DataFrame,
    note_table: pd.DataFrame,
    melodic_sequences: dict[str, list[dict[str, Any]]],
    global_key: m21key.Key | None,
) -> pd.DataFrame:
    columns = [
        "cadence_type",
        "measure_number",
        "beat",
        "strength_label",
        "melody_part",
        "melody_pitch_name",
        "melody_scale_degree",
        "melody_skeleton_class",
        "previous_measure_number",
        "previous_beat",
        "previous_roman_numeral",
        "previous_bass_pitch_name",
        "previous_bass_scale_degree",
        "current_roman_numeral",
        "current_bass_pitch_name",
        "current_bass_scale_degree",
        "cadence_window",
        "candidate_score",
        "dominant_score",
        "tonic_score",
        "melody_score",
        "bass_score",
        "closure_score",
        "cadence_key",
        "global_key",
    ]
    if harmony_table.empty:
        return pd.DataFrame(columns=columns)

    melody_part_name, primary_melodic_events = _select_primary_melodic_events(melodic_sequences)
    if not primary_melodic_events:
        return pd.DataFrame(columns=columns)
    measure_lowest_map = _measure_lowest_pitch_map(note_table)
    opening_lowest_map = _measure_window_lowest_pitch_map(note_table, max_beat=2.0)
    closing_lowest_map = _measure_window_lowest_pitch_map(note_table, min_beat=2.0)
    measure_rows_map = _group_rows_by_measure(harmony_table)
    primary_events_by_measure = _group_melodic_events_by_measure(primary_melodic_events)

    rows: list[dict[str, Any]] = []
    for current_measure in sorted(measure_rows_map):
        if current_measure <= min(measure_rows_map):
            continue

        for cadence_key in _candidate_cadence_keys(
            current_measure=current_measure,
            measure_rows_map=measure_rows_map,
            primary_events_by_measure=primary_events_by_measure,
            opening_lowest_map=opening_lowest_map,
            global_key=global_key,
        ):
            is_global_key_candidate = (
                global_key is not None
                and format_key_label(cadence_key) == format_key_label(global_key)
            )
            harmonic_tonic_support = _score_tonic_arrival(current_measure, cadence_key, measure_rows_map)
            melodic_tonic_support = _score_tonic_line_arrival(current_measure, cadence_key, primary_events_by_measure)
            tonic_support = (
                melodic_tonic_support
                if float(melodic_tonic_support["score"]) > float(harmonic_tonic_support["score"])
                else harmonic_tonic_support
            )
            if tonic_support["score"] < 0.45:
                continue

            arrival_measure_number = _safe_int(tonic_support.get("measure_number"), current_measure)
            if arrival_measure_number not in {current_measure, current_measure + 1}:
                continue

            harmonic_dominant_support = _score_dominant_preparation(arrival_measure_number, cadence_key, measure_rows_map)
            melodic_preparation_support = _score_melodic_preparation(
                arrival_measure_number,
                cadence_key,
                primary_events_by_measure,
            )
            dominant_support = (
                melodic_preparation_support
                if float(melodic_preparation_support["score"]) > float(harmonic_dominant_support["score"])
                else harmonic_dominant_support
            )
            if dominant_support["score"] < 0.35:
                continue
            dominant_is_harmonic = _roman_is_dominant(str(harmonic_dominant_support["roman_numeral"]))
            dominant_is_melodic = str(dominant_support["roman_numeral"]) == "V（旋律准备）"
            if not dominant_is_harmonic and not dominant_is_melodic:
                continue

            previous_measure_number = _safe_int(dominant_support.get("measure_number"), arrival_measure_number - 1)
            bass_support = _score_bass_support(
                previous_measure=previous_measure_number,
                current_measure=arrival_measure_number,
                cadence_key=cadence_key,
                measure_lowest_map=measure_lowest_map,
                opening_lowest_map=opening_lowest_map,
                closing_lowest_map=closing_lowest_map,
            )
            melody_support = _score_melodic_skeleton(arrival_measure_number, cadence_key, primary_events_by_measure)
            closure_support = _score_cadential_closure(
                current_measure=arrival_measure_number,
                previous_measure=previous_measure_number,
                cadence_key=cadence_key,
                measure_rows_map=measure_rows_map,
                melody_event=melody_support.get("event"),
                tonic_support=tonic_support,
            )
            if bass_support["score"] < 0.20:
                continue
            allow_restored_dominant_bass = (
                bass_support["previous_degree"] in {None, 7}
                and bass_support["current_degree"] == 1
                and float(dominant_support["score"]) >= 0.55
                and (dominant_is_harmonic or dominant_is_melodic)
            )
            allow_line_based_tonic_arrival = (
                float(tonic_support["score"]) >= 0.75
                and float(dominant_support["score"]) >= 0.50
                and melody_support.get("scale_degree") == 1
                and is_global_key_candidate
            )
            allow_reduced_bass_pac = (
                melody_support.get("scale_degree") == 1
                and float(tonic_support["score"]) >= 0.75
                and float(dominant_support["score"]) >= 0.75
                and float(closure_support["score"]) >= 0.55
                and bass_support["previous_degree"] in {2, 5, 7, None}
                and bass_support["current_degree"] in {1, 5, None}
            )
            if bass_support["current_degree"] != 1 and not allow_line_based_tonic_arrival and not allow_reduced_bass_pac:
                continue
            if (
                bass_support["previous_degree"] != 5
                and not allow_restored_dominant_bass
                and not allow_line_based_tonic_arrival
                and not allow_reduced_bass_pac
            ):
                continue

            if melody_support.get("scale_degree") != 1:
                continue
            tonic_is_literal = _roman_is_literal_tonic_arrival(str(harmonic_tonic_support["roman_numeral"]))
            tonic_is_melodic = str(tonic_support["roman_numeral"]) == _display_tonic_label_for_key(cadence_key)
            if not tonic_is_literal and not tonic_is_melodic:
                continue
            if not is_global_key_candidate and not tonic_is_literal:
                local_resolution_ready = (
                    float(harmonic_dominant_support["score"]) >= 0.70
                    or melodic_preparation_support.get("previous_degree") in {2, 7}
                    or bass_support["previous_degree"] in {5, 7}
                )
                if not local_resolution_ready:
                    continue
            if global_key is not None and not is_global_key_candidate:
                if not tonic_is_literal and float(melodic_preparation_support["score"]) < 0.35:
                    continue

            allow_sparse_closure = (
                float(tonic_support["score"]) >= 0.75
                and float(dominant_support["score"]) >= 0.50
                and float(closure_support["score"]) >= 0.18
            )
            if float(closure_support["score"]) < 0.58 and not allow_sparse_closure:
                continue

            candidate_score = (
                float(dominant_support["score"]) * 1.20
                + float(tonic_support["score"]) * 1.35
                + float(melody_support["score"]) * 1.00
                + float(bass_support["score"]) * 0.95
                + float(closure_support["score"]) * 0.75
            )
            if candidate_score < 2.70:
                continue

            melody_event = melody_support.get("event")
            melody_pitch_name = str(melody_event.get("pitch_name", "")) if melody_event is not None else ""
            melody_scale_degree = melody_support.get("scale_degree")
            rows.append(
                {
                    "cadence_type": "完满终止候选",
                    "measure_number": arrival_measure_number,
                    "beat": round(
                        float(tonic_support["beat"]) if arrival_measure_number == _safe_int(tonic_support["measure_number"], arrival_measure_number) else 1.0,
                        3,
                    ),
                    "strength_label": str(closure_support["strength_label"]),
                    "melody_part": str(melody_part_name or ""),
                    "melody_pitch_name": melody_pitch_name,
                    "melody_scale_degree": int(melody_scale_degree) if melody_scale_degree is not None else None,
                    "melody_skeleton_class": _melody_skeleton_class_label(
                        int(melody_scale_degree) if melody_scale_degree is not None else None
                    ),
                    "previous_measure_number": previous_measure_number,
                    "previous_beat": round(_safe_float(dominant_support.get("beat"), 0.0), 3),
                    "previous_roman_numeral": str(dominant_support["roman_numeral"]),
                    "previous_bass_pitch_name": str(bass_support["previous_pitch_name"]),
                    "previous_bass_scale_degree": bass_support["previous_degree"],
                    "current_roman_numeral": str(tonic_support["roman_numeral"]),
                    "current_bass_pitch_name": str(bass_support["current_pitch_name"]),
                    "current_bass_scale_degree": bass_support["current_degree"],
                    "cadence_window": f"{previous_measure_number}-{arrival_measure_number}",
                    "candidate_score": round(candidate_score, 3),
                    "dominant_score": round(float(dominant_support["score"]), 3),
                    "tonic_score": round(float(tonic_support["score"]), 3),
                    "melody_score": round(float(melody_support["score"]), 3),
                    "bass_score": round(float(bass_support["score"]), 3),
                    "closure_score": round(float(closure_support["score"]), 3),
                    "cadence_key": format_key_label(cadence_key),
                    "global_key": format_key_label(global_key),
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)
    cadence_table = pd.DataFrame(rows)
    cadence_table = (
        cadence_table.sort_values(["measure_number", "candidate_score", "beat"], ascending=[True, False, True])
        .drop_duplicates(subset=["measure_number"], keep="first")
        .reset_index(drop=True)
    )
    if len(cadence_table) >= 8 and cadence_table["cadence_key"].astype(str).nunique() >= 3:
        global_key_label = format_key_label(global_key)
        key_summary = (
            cadence_table.groupby("cadence_key", as_index=False)
            .agg(candidate_count=("cadence_key", "size"), score_sum=("candidate_score", "sum"))
        )
        retained_keys = set(
            key_summary.loc[
                (key_summary["candidate_count"] >= 2) | (key_summary["score_sum"] >= 10.0),
                "cadence_key",
            ].astype(str)
        )
        retained_keys.add(global_key_label)
        if retained_keys:
            cadence_table = cadence_table.loc[cadence_table["cadence_key"].astype(str).isin(retained_keys)].reset_index(drop=True)
    return cadence_table


def _score_dominant_arrival(
    current_measure: int,
    cadence_key: m21key.Key,
    measure_rows_map: dict[int, pd.DataFrame],
) -> dict[str, Any]:
    current_rows = measure_rows_map.get(current_measure, pd.DataFrame())
    previous_closing = _slice_measure_frame(measure_rows_map.get(current_measure - 1), min_beat=2.0)
    dominant_pc_sets = [
        _roman_pitch_class_set("V", cadence_key),
        _roman_pitch_class_set("V7", cadence_key),
    ]

    best_score = 0.0
    best_row: pd.Series | None = None
    best_rn = ""

    for frame in [current_rows, previous_closing]:
        if frame is None or frame.empty:
            continue
        for _, row in frame.iterrows():
            row_pitch_classes = _frame_pitch_classes(pd.DataFrame([row]))
            dominant_coverage = max(
                (_pitch_class_coverage(row_pitch_classes, pcs) for pcs in dominant_pc_sets if pcs),
                default=0.0,
            )
            row_chord = _safe_chord_from_pitch_names(row.get("pitch_names", ""))
            row_rn = _safe_roman_figure_for_key(row_chord, cadence_key)

            row_score = 0.0
            if _roman_is_dominant(row_rn):
                row_score = 0.44 if len(row_pitch_classes) == 1 else 0.78
            elif dominant_coverage >= (2.0 / 3.0):
                row_score = 0.34 + 0.30 * dominant_coverage

            if row_score <= 0.0:
                continue

            beat = _safe_float(row.get("beat"), 0.0)
            duration = _safe_float(row.get("quarter_length"), 0.0)
            if _is_strong_beat(beat):
                row_score += 0.12
            elif beat <= 2.0:
                row_score += 0.06
            if duration >= 1.0:
                row_score += 0.12
            elif duration >= 0.5:
                row_score += 0.05
            if _safe_int(row.get("measure_number"), current_measure) == current_measure:
                row_score += 0.10

            if row_score > best_score:
                best_score = row_score
                best_row = row
                best_rn = row_rn

    current_pitch_classes = _frame_pitch_classes(current_rows)
    dominant_window_coverage = max(
        (_pitch_class_coverage(current_pitch_classes, pcs) for pcs in dominant_pc_sets if pcs),
        default=0.0,
    )
    if dominant_window_coverage >= 1.0:
        best_score += 0.18
    elif dominant_window_coverage >= (2.0 / 3.0):
        best_score += 0.10

    if best_row is None or best_score <= 0.0:
        return {
            "score": 0.0,
            "roman_numeral": "",
            "measure_number": current_measure,
            "beat": 1.0,
            "quarter_length": 0.0,
        }

    return {
        "score": min(best_score, 1.15),
        "roman_numeral": best_rn or "V（和声还原）",
        "measure_number": _safe_int(best_row.get("measure_number"), current_measure),
        "beat": round(_safe_float(best_row.get("beat"), 1.0), 3),
        "quarter_length": round(_safe_float(best_row.get("quarter_length"), 0.0), 3),
    }


def _score_half_cadence_melodic_goal(
    current_measure: int,
    cadence_key: m21key.Key,
    primary_events_by_measure: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    candidate_events = list(primary_events_by_measure.get(current_measure, []))
    if not candidate_events:
        return {
            "score": 0.0,
            "event": None,
            "scale_degree": None,
        }

    best_score = 0.0
    best_event: dict[str, Any] | None = None
    best_degree: int | None = None
    for event in candidate_events:
        pitch_object = _pitch_from_midi(_safe_int(event.get("midi"), 0))
        degree = _safe_scale_degree(cadence_key, pitch_object)
        if degree not in {2, 5, 7}:
            continue

        score = 0.0
        if degree == 5:
            score = 0.76
        elif degree == 2:
            score = 0.68
        elif degree == 7:
            score = 0.58

        beat = _safe_float(event.get("beat"), 0.0)
        duration = _safe_float(event.get("quarter_length"), 0.0)
        if _is_strong_beat(beat):
            score += 0.12
        elif beat <= 2.0:
            score += 0.06
        if duration >= 1.0:
            score += 0.12
        elif duration >= 0.5:
            score += 0.05

        if score > best_score:
            best_score = score
            best_event = event
            best_degree = degree

    if best_event is None:
        return {
            "score": 0.0,
            "event": None,
            "scale_degree": None,
        }

    return {
        "score": min(best_score, 0.95),
        "event": best_event,
        "scale_degree": best_degree,
    }


def _score_half_cadence_bass_support(
    current_measure: int,
    cadence_key: m21key.Key,
    measure_lowest_map: dict[int, dict[str, Any]],
    opening_lowest_map: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    current_primary = opening_lowest_map.get(current_measure)
    current_fallback = measure_lowest_map.get(current_measure)

    best_score = 0.0
    best_degree = None
    best_pitch_name = ""
    for info, penalty in [(current_primary, 0.0), (current_fallback, 0.05)]:
        if info is None:
            continue
        pitch_object = _pitch_from_midi(int(info["midi"]))
        degree = _safe_scale_degree(cadence_key, pitch_object)
        score = 0.0
        if degree == 5:
            score = 0.52 - penalty
        elif degree == 7:
            score = 0.24 - penalty
        elif degree == 2:
            score = 0.14 - penalty
        if score > best_score:
            best_score = score
            best_degree = degree
            best_pitch_name = str(info["pitch_name"])

    return {
        "score": min(best_score, 0.60),
        "current_pitch_name": best_pitch_name,
        "current_degree": best_degree,
    }


def _score_half_cadence_closure(
    current_measure: int,
    measure_rows_map: dict[int, pd.DataFrame],
    melody_event: dict[str, Any] | None,
    dominant_support: dict[str, Any],
) -> dict[str, Any]:
    score = 0.0
    current_frame = measure_rows_map.get(current_measure, pd.DataFrame())
    dominant_measure = _safe_int(dominant_support.get("measure_number"), current_measure)
    dominant_beat = _safe_float(dominant_support.get("beat"), 1.0)
    dominant_duration = _safe_float(dominant_support.get("quarter_length"), 0.0)

    if dominant_measure == current_measure:
        score += 0.12
    if dominant_beat <= 1.5:
        score += 0.12
    elif dominant_beat <= 2.5:
        score += 0.06
    if dominant_duration >= 1.0:
        score += 0.16
    elif dominant_duration >= 0.5:
        score += 0.08
    if melody_event is not None:
        melody_beat = _safe_float(melody_event.get("beat"), 0.0)
        melody_duration = _safe_float(melody_event.get("quarter_length"), 0.0)
        if melody_beat <= 1.5:
            score += 0.08
        if melody_duration >= 1.0:
            score += 0.16
        elif melody_duration >= 0.5:
            score += 0.08
    if not current_frame.empty and current_frame["quarter_length"].astype(float).max() >= 1.0:
        score += 0.10

    if score >= 0.50:
        strength_label = "高分候选"
    elif score >= 0.32:
        strength_label = "中高候选"
    else:
        strength_label = "中等候选"

    return {
        "score": min(score, 0.75),
        "strength_label": strength_label,
    }


def _build_half_cadence_candidates(
    harmony_table: pd.DataFrame,
    note_table: pd.DataFrame,
    melodic_sequences: dict[str, list[dict[str, Any]]],
    global_key: m21key.Key | None,
) -> pd.DataFrame:
    columns = [
        "cadence_type",
        "measure_number",
        "beat",
        "strength_label",
        "melody_part",
        "melody_pitch_name",
        "melody_scale_degree",
        "melody_skeleton_class",
        "previous_measure_number",
        "previous_beat",
        "previous_roman_numeral",
        "previous_bass_pitch_name",
        "previous_bass_scale_degree",
        "current_roman_numeral",
        "current_bass_pitch_name",
        "current_bass_scale_degree",
        "cadence_window",
        "candidate_score",
        "dominant_score",
        "tonic_score",
        "melody_score",
        "bass_score",
        "closure_score",
        "cadence_key",
        "global_key",
    ]
    if harmony_table.empty or global_key is None:
        return pd.DataFrame(columns=columns)

    melody_part_name, primary_melodic_events = _select_primary_melodic_events(melodic_sequences)
    if not primary_melodic_events:
        return pd.DataFrame(columns=columns)

    cadence_key = global_key
    measure_lowest_map = _measure_lowest_pitch_map(note_table)
    opening_lowest_map = _measure_window_lowest_pitch_map(note_table, max_beat=2.0)
    measure_rows_map = _group_rows_by_measure(harmony_table)
    primary_events_by_measure = _group_melodic_events_by_measure(primary_melodic_events)

    rows: list[dict[str, Any]] = []
    for current_measure in sorted(measure_rows_map):
        if current_measure <= min(measure_rows_map):
            continue

        dominant_support = _score_dominant_arrival(current_measure, cadence_key, measure_rows_map)
        if float(dominant_support["score"]) < 0.58:
            continue
        if not _roman_is_dominant(str(dominant_support["roman_numeral"])):
            continue

        melody_support = _score_half_cadence_melodic_goal(current_measure, cadence_key, primary_events_by_measure)
        if float(melody_support["score"]) < 0.52:
            continue

        bass_support = _score_half_cadence_bass_support(
            current_measure=current_measure,
            cadence_key=cadence_key,
            measure_lowest_map=measure_lowest_map,
            opening_lowest_map=opening_lowest_map,
        )
        if float(bass_support["score"]) < 0.28:
            continue

        tonic_support = _score_tonic_arrival(current_measure, cadence_key, measure_rows_map)
        tonic_arrives_inside_current_measure = _safe_int(tonic_support.get("measure_number"), current_measure) == current_measure
        if (
            tonic_arrives_inside_current_measure
            and float(tonic_support["score"]) >= 0.82
            and _roman_is_literal_tonic_arrival(str(tonic_support["roman_numeral"]))
        ):
            continue

        closure_support = _score_half_cadence_closure(
            current_measure=current_measure,
            measure_rows_map=measure_rows_map,
            melody_event=melody_support.get("event"),
            dominant_support=dominant_support,
        )
        if float(closure_support["score"]) < 0.28:
            continue

        candidate_score = (
            float(dominant_support["score"]) * 1.35
            + float(melody_support["score"]) * 0.95
            + float(bass_support["score"]) * 1.00
            + float(closure_support["score"]) * 0.85
        )
        if candidate_score < 2.20:
            continue

        melody_event = melody_support.get("event")
        melody_pitch_name = str(melody_event.get("pitch_name", "")) if melody_event is not None else ""
        melody_scale_degree = melody_support.get("scale_degree")
        rows.append(
            {
                "cadence_type": "半终止候选",
                "measure_number": current_measure,
                "beat": round(float(dominant_support["beat"]), 3),
                "strength_label": str(closure_support["strength_label"]),
                "melody_part": str(melody_part_name or ""),
                "melody_pitch_name": melody_pitch_name,
                "melody_scale_degree": int(melody_scale_degree) if melody_scale_degree is not None else None,
                "melody_skeleton_class": _melody_skeleton_class_label(
                    int(melody_scale_degree) if melody_scale_degree is not None else None
                ),
                "previous_measure_number": current_measure - 1,
                "previous_beat": round(max(1.0, float(dominant_support["beat"]) - 1.0), 3),
                "previous_roman_numeral": "",
                "previous_bass_pitch_name": "",
                "previous_bass_scale_degree": None,
                "current_roman_numeral": str(dominant_support["roman_numeral"]),
                "current_bass_pitch_name": str(bass_support["current_pitch_name"]),
                "current_bass_scale_degree": bass_support["current_degree"],
                "cadence_window": f"{current_measure - 1}-{current_measure}",
                "candidate_score": round(candidate_score, 3),
                "dominant_score": round(float(dominant_support["score"]), 3),
                "tonic_score": round(float(tonic_support["score"]), 3),
                "melody_score": round(float(melody_support["score"]), 3),
                "bass_score": round(float(bass_support["score"]), 3),
                "closure_score": round(float(closure_support["score"]), 3),
                "cadence_key": format_key_label(cadence_key),
                "global_key": format_key_label(global_key),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .sort_values(["measure_number", "candidate_score", "beat"], ascending=[True, False, True])
        .drop_duplicates(subset=["measure_number"], keep="first")
        .reset_index(drop=True)
    )


def _build_cadence_candidates(
    harmony_table: pd.DataFrame,
    note_table: pd.DataFrame,
    melodic_sequences: dict[str, list[dict[str, Any]]],
    global_key: m21key.Key | None,
) -> pd.DataFrame:
    pac_candidates = _build_perfect_authentic_cadence_candidates(
        harmony_table=harmony_table,
        note_table=note_table,
        melodic_sequences=melodic_sequences,
        global_key=global_key,
    )
    hc_candidates = _build_half_cadence_candidates(
        harmony_table=harmony_table,
        note_table=note_table,
        melodic_sequences=melodic_sequences,
        global_key=global_key,
    )

    if pac_candidates.empty and hc_candidates.empty:
        return pac_candidates

    if not pac_candidates.empty and not hc_candidates.empty and global_key is not None:
        global_key_label = format_key_label(global_key)
        suppress_indices: list[int] = []
        for index, row in pac_candidates.iterrows():
            same_measure_hc = hc_candidates.loc[
                (hc_candidates["measure_number"].astype(int) == int(row["measure_number"]))
                & (hc_candidates["cadence_key"].astype(str) == global_key_label)
            ]
            if same_measure_hc.empty:
                continue
            if str(row["cadence_key"]) == global_key_label:
                continue
            previous_bass_degree = _safe_int(row.get("previous_bass_scale_degree"), None)
            hc_score = float(same_measure_hc["candidate_score"].astype(float).max())
            pac_score = float(row["candidate_score"])
            if previous_bass_degree not in {5, 7} and hc_score >= pac_score - 0.80:
                suppress_indices.append(index)
        if suppress_indices:
            pac_candidates = pac_candidates.drop(index=suppress_indices).reset_index(drop=True)

    frames = [frame for frame in [pac_candidates, hc_candidates] if not frame.empty]
    if not frames:
        return pac_candidates
    combined = pd.concat([frame.astype(object) for frame in frames], ignore_index=True)
    if combined.empty:
        return combined
    return combined.sort_values(
        ["measure_number", "cadence_type", "candidate_score", "beat"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def _rhythm_signature(durations: list[float]) -> tuple[float, ...]:
    total = float(sum(durations))
    if total <= 1e-8:
        return tuple(0.0 for _ in durations)
    return tuple(round(float(duration) / total, 3) for duration in durations)


def _window_excerpt(events: list[dict[str, Any]]) -> str:
    return " ".join(str(event["pitch_name"]) for event in events)


def _window_is_thematically_informative(pitches: list[int], interval_signature: tuple[int, ...]) -> bool:
    unique_pitches = len(set(int(value) for value in pitches))
    if unique_pitches <= 1:
        return False
    if interval_signature and all(int(value) == 0 for value in interval_signature):
        return False
    return True


def _is_close_to_value(left: float, right: float, tolerance: float = 0.05) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _theme_start_priority(events: list[dict[str, Any]], start_index: int) -> int:
    current_event = events[start_index]
    current_measure = int(current_event["measure_number"])
    current_beat = float(current_event["beat"])
    current_offset = float(current_event["offset_ql"])

    priority = 0
    if start_index == 0:
        priority += 3
    else:
        previous_event = events[start_index - 1]
        previous_measure = int(previous_event["measure_number"])
        previous_offset = float(previous_event["offset_ql"])
        previous_duration = float(previous_event["quarter_length"])
        if current_measure != previous_measure:
            priority += 3
        if current_offset > previous_offset + previous_duration + 1e-6:
            priority += 1

    if _is_close_to_value(current_beat, 1.0):
        priority += 3
    elif _is_close_to_value(current_beat, 3.0):
        priority += 2
    elif _is_close_to_value(current_beat, round(current_beat)):
        priority += 1

    return priority


def _match_priority_score(
    anchor: dict[str, Any],
    match: dict[str, Any],
    relation_type: str,
    similarity_score: float,
) -> float:
    measure_distance = abs(int(match["measure_number"]) - int(anchor["measure_number"]))
    anchor_priority = int(anchor.get("start_priority", 0))
    match_priority = int(match.get("start_priority", 0))

    start_bonus = 0.04 * min(anchor_priority, 6) + 0.04 * min(match_priority, 6)
    locality_bonus = 0.0
    if measure_distance <= 2:
        locality_bonus = 0.28
    elif measure_distance <= 4:
        locality_bonus = 0.18
    elif measure_distance <= 8:
        locality_bonus = 0.08

    sequence_bonus = 0.0
    if relation_type in {"移调再现", "轮廓相似"} and 1 <= measure_distance <= 4:
        sequence_bonus = 0.30

    return float(similarity_score) + start_bonus + locality_bonus + sequence_bonus


def _window_occurrences(
    melodic_sequences: dict[str, list[dict[str, Any]]],
    window_size: int,
) -> list[dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []
    for part_name, events in melodic_sequences.items():
        if len(events) < window_size:
            continue
        for start_index in range(0, len(events) - window_size + 1):
            window = events[start_index : start_index + window_size]
            pitches = [int(event["midi"]) for event in window]
            durations = [float(event["quarter_length"]) for event in window]
            interval_signature = tuple(pitches[index + 1] - pitches[index] for index in range(len(pitches) - 1))
            if not _window_is_thematically_informative(pitches, interval_signature):
                continue
            contour_signature = tuple(int(np.sign(value)) for value in interval_signature)
            start_priority = _theme_start_priority(events, start_index)
            occurrences.append(
                {
                    "part_name": part_name,
                    "start_index": start_index,
                    "measure_number": int(window[0]["measure_number"]),
                    "beat": float(window[0]["beat"]),
                    "start_priority": start_priority,
                    "pitch_signature": tuple(pitches),
                    "interval_signature": interval_signature,
                    "contour_signature": contour_signature,
                    "duration_signature": tuple(round(duration, 3) for duration in durations),
                    "rhythm_signature": _rhythm_signature(durations),
                    "first_pitch": pitches[0],
                    "excerpt": _window_excerpt(window),
                }
            )
    return occurrences


def _non_overlapping(
    left: dict[str, Any],
    right: dict[str, Any],
    window_size: int,
) -> bool:
    if left["part_name"] != right["part_name"]:
        return True
    return abs(int(left["start_index"]) - int(right["start_index"])) >= window_size


def _dedupe_occurrences(occurrences: list[dict[str, Any]], window_size: int) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for occurrence in sorted(occurrences, key=lambda item: (str(item["part_name"]), int(item["start_index"]))):
        if all(_non_overlapping(occurrence, existing, window_size) for existing in kept):
            kept.append(occurrence)
    return kept


def _build_theme_matches(
    melodic_sequences: dict[str, list[dict[str, Any]]],
    window_size: int,
    max_results: int,
) -> pd.DataFrame:
    if window_size < 3:
        window_size = 3

    occurrences = _window_occurrences(melodic_sequences, window_size)
    if not occurrences:
        return pd.DataFrame(
            columns=[
                "relation_type",
                "similarity_score",
                "source_part",
                "source_measure",
                "source_beat",
                "source_excerpt",
                "match_part",
                "match_measure",
                "match_beat",
                "match_excerpt",
                "transposition_semitones",
                "window_size",
                "match_detail",
            ]
        )

    exact_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    transposed_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    contour_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

    for occurrence in occurrences:
        exact_key = (occurrence["pitch_signature"], occurrence["duration_signature"])
        transposed_key = (occurrence["interval_signature"], occurrence["rhythm_signature"])
        contour_key = (occurrence["contour_signature"], occurrence["rhythm_signature"])
        exact_groups.setdefault(exact_key, []).append(occurrence)
        transposed_groups.setdefault(transposed_key, []).append(occurrence)
        contour_groups.setdefault(contour_key, []).append(occurrence)

    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[tuple[str, int], tuple[str, int]]] = set()

    def add_group(
        relation_type: str,
        groups: dict[tuple[Any, ...], list[dict[str, Any]]],
        similarity_score: float,
        match_detail: str,
    ) -> None:
        for group_occurrences in groups.values():
            deduped = _dedupe_occurrences(group_occurrences, window_size)
            deduped = sorted(
                deduped,
                key=lambda item: (
                    -int(item.get("start_priority", 0)),
                    int(item["measure_number"]),
                    float(item["beat"]),
                    int(item["start_index"]),
                ),
            )
            preferred = [item for item in deduped if int(item.get("start_priority", 0)) >= 4]
            if len(preferred) >= 2:
                deduped = preferred
            if len(deduped) < 2:
                continue
            anchor = deduped[0]
            for match in deduped[1:]:
                pair_key = tuple(
                    sorted(
                        [
                            (str(anchor["part_name"]), int(anchor["start_index"])),
                            (str(match["part_name"]), int(match["start_index"])),
                        ]
                    )
                )
                if pair_key in seen_pairs:
                    continue
                if relation_type == "移调再现" and anchor["pitch_signature"] == match["pitch_signature"]:
                    continue
                if relation_type == "轮廓相似" and anchor["interval_signature"] == match["interval_signature"]:
                    continue

                seen_pairs.add(pair_key)
                left = anchor
                right = match
                left_position = (int(left["measure_number"]), float(left["beat"]), int(left["start_index"]))
                right_position = (int(right["measure_number"]), float(right["beat"]), int(right["start_index"]))
                if right_position < left_position:
                    left, right = right, left

                priority_score = _match_priority_score(left, right, relation_type, similarity_score)
                rows.append(
                    {
                        "relation_type": relation_type,
                        "similarity_score": similarity_score,
                        "priority_score": priority_score,
                        "source_part": str(left["part_name"]),
                        "source_measure": int(left["measure_number"]),
                        "source_beat": round(float(left["beat"]), 3),
                        "source_excerpt": str(left["excerpt"]),
                        "match_part": str(right["part_name"]),
                        "match_measure": int(right["measure_number"]),
                        "match_beat": round(float(right["beat"]), 3),
                        "match_excerpt": str(right["excerpt"]),
                        "transposition_semitones": int(right["first_pitch"]) - int(left["first_pitch"]),
                        "window_size": window_size,
                        "match_detail": match_detail,
                    }
                )

    add_group("精确再现", exact_groups, 1.00, "音高与节奏窗口完全一致")
    add_group("移调再现", transposed_groups, 0.90, "音程骨架与节奏比例一致，但起始音高不同")
    add_group("轮廓相似", contour_groups, 0.72, "上下行轮廓与节奏比例一致，但具体音程发生变化")

    if not rows:
        return pd.DataFrame(
            columns=[
                "relation_type",
                "similarity_score",
                "source_part",
                "source_measure",
                "source_beat",
                "source_excerpt",
                "match_part",
                "match_measure",
                "match_beat",
                "match_excerpt",
                "transposition_semitones",
                "window_size",
                "match_detail",
            ]
        )

    theme_matches = pd.DataFrame(rows)
    theme_matches = theme_matches.sort_values(
        ["priority_score", "similarity_score", "source_measure", "match_measure", "source_beat", "match_beat"],
        ascending=[False, False, True, True, True, True],
    ).reset_index(drop=True)
    return theme_matches.drop(columns=["priority_score"]).head(max_results)


def _top_histogram_statement(
    histogram: pd.DataFrame,
    label_column: str,
    count_column: str = "count",
    limit: int = 3,
) -> str:
    if histogram.empty:
        return "暂无"
    top = histogram.sort_values(count_column, ascending=False).head(limit)
    return " / ".join(f"{row[label_column]}×{int(row[count_column])}" for _, row in top.iterrows())


def _build_summary_lines(
    global_key: m21key.Key | None,
    total_notes: int,
    unique_pitches: int,
    unique_pitch_classes: int,
    pitch_class_histogram: pd.DataFrame,
    interval_class_histogram: pd.DataFrame,
    vertical_interval_histogram: pd.DataFrame,
    harmony_table: pd.DataFrame,
    cadence_candidates: pd.DataFrame,
    theme_matches: pd.DataFrame,
    theme_window_notes: int,
) -> list[str]:
    roman_count = 0
    if not harmony_table.empty and "roman_numeral" in harmony_table.columns:
        roman_count = int(harmony_table["roman_numeral"].astype(str).str.strip().ne("").sum())

    pac_count = 0
    hc_count = 0
    if not cadence_candidates.empty and "cadence_type" in cadence_candidates.columns:
        pac_count = int(cadence_candidates["cadence_type"].astype(str).eq("完满终止候选").sum())
        hc_count = int(cadence_candidates["cadence_type"].astype(str).eq("半终止候选").sum())

    return [
        f"全局调性 / 调式估计：{format_key_label(global_key)}",
        f"共提取 {total_notes} 个音高事件，包含 {unique_pitches} 个不同音高、{unique_pitch_classes} 个不同音级类。",
        f"音级类重心：{_top_histogram_statement(pitch_class_histogram, 'pitch_class_label')}",
        f"主导音程序类：{_top_histogram_statement(interval_class_histogram, 'interval_class')}",
        f"主导纵向音程：{_top_histogram_statement(vertical_interval_histogram, 'interval_name')}",
        f"共生成 {len(harmony_table)} 个和声切片，其中 {roman_count} 个切片得到了 Roman numeral 候选。",
        (
            f"共检测到 {len(cadence_candidates)} 个终止候选，其中完满终止 {pac_count} 个、半终止 {hc_count} 个。"
            "当前采用候选评分：综合属准备 / 属到达、旋律骨架、终止窗口低音与句末收束。"
        ),
        (
            f"基于连续 {theme_window_notes} 音窗口，共找到 {len(theme_matches)} 条主题 / 动机再现候选。"
            "当前结果属于第一阶段辅助匹配，需要研究者继续复核。"
        ),
    ]


def build_symbolic_export_payload(
    result: dict[str, Any],
    harmony_annotations: pd.DataFrame | None = None,
    cadence_annotations: pd.DataFrame | None = None,
    theme_annotations: pd.DataFrame | None = None,
) -> bytes:
    payload = {
        "source_name": result["source_name"],
        "score_title": result["score_title"],
        "config": result["config"],
        "global_key": result["global_key"],
        "summary_lines": result["summary_lines"],
        "part_summary": result["part_summary"].to_dict(orient="records"),
        "note_table": result["note_table"].to_dict(orient="records"),
        "pitch_class_histogram": result["pitch_class_histogram"].to_dict(orient="records"),
        "pitch_height_histogram": result["pitch_height_histogram"].to_dict(orient="records"),
        "scale_degree_histogram": result["scale_degree_histogram"].to_dict(orient="records"),
        "measure_pitch_summary": result["measure_pitch_summary"].to_dict(orient="records"),
        "interval_table": result["interval_table"].to_dict(orient="records"),
        "interval_class_histogram": result["interval_class_histogram"].to_dict(orient="records"),
        "directed_interval_histogram": result["directed_interval_histogram"].to_dict(orient="records"),
        "vertical_interval_table": result["vertical_interval_table"].to_dict(orient="records"),
        "vertical_interval_histogram": result["vertical_interval_histogram"].to_dict(orient="records"),
        "measure_consonance_summary": result["measure_consonance_summary"].to_dict(orient="records"),
        "harmony_table": (harmony_annotations if harmony_annotations is not None else result["harmony_table"]).to_dict(
            orient="records"
        ),
        "cadence_candidates": (
            cadence_annotations if cadence_annotations is not None else result["cadence_candidates"]
        ).to_dict(orient="records"),
        "theme_matches": (theme_annotations if theme_annotations is not None else result["theme_matches"]).to_dict(
            orient="records"
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def export_symbolic_score_file(
    source: str | Path | BinaryIO | m21stream.Score,
    fmt: str = "musicxml",
) -> tuple[bytes, str, str]:
    score, source_name = _load_symbolic_score(source)
    stem = Path(source_name).stem or "score_input"

    suffix_map = {
        "musicxml": ".musicxml",
        "xml": ".xml",
        "mxl": ".mxl",
    }
    mime_map = {
        "musicxml": "application/vnd.recordare.musicxml+xml",
        "xml": "application/xml",
        "mxl": "application/vnd.recordare.musicxml",
    }
    suffix = suffix_map.get(fmt, ".musicxml")
    mime_type = mime_map.get(fmt, "application/octet-stream")

    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temporary_path = temporary.name
    temporary.close()
    try:
        score.write(fmt, fp=temporary_path)
        data = Path(temporary_path).read_bytes()
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)

    return data, f"{stem}{suffix}", mime_type
