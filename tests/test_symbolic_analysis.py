from __future__ import annotations

import unittest

from music21 import chord, key, meter, note, stream

from spectral_tool.symbolic_analysis import SymbolicAnalysisConfig, analyze_symbolic_score, export_symbolic_score_file


def _build_exact_recurrence_score() -> stream.Score:
    score = stream.Score(id="ExactRecurrence")
    melody = stream.Part(id="Melody")
    melody.partName = "Melody"
    melody.insert(0, key.Key("C"))
    melody.insert(0, meter.TimeSignature("4/4"))

    measure_1 = stream.Measure(number=1)
    for pitch_name in ["C4", "D4", "E4", "G4"]:
        measure_1.append(note.Note(pitch_name, quarterLength=1.0))

    measure_2 = stream.Measure(number=2)
    for pitch_name in ["C4", "D4", "E4", "G4"]:
        measure_2.append(note.Note(pitch_name, quarterLength=1.0))

    melody.append(measure_1)
    melody.append(measure_2)
    score.append(melody)
    return score


def _build_transposed_recurrence_score() -> stream.Score:
    score = stream.Score(id="TransposedRecurrence")
    melody = stream.Part(id="Melody")
    melody.partName = "Melody"
    melody.insert(0, key.Key("G"))
    melody.insert(0, meter.TimeSignature("4/4"))

    measure_1 = stream.Measure(number=1)
    for pitch_name in ["G4", "A4", "B4", "D5"]:
        measure_1.append(note.Note(pitch_name, quarterLength=1.0))

    measure_2 = stream.Measure(number=2)
    for pitch_name in ["A4", "B4", "C#5", "E5"]:
        measure_2.append(note.Note(pitch_name, quarterLength=1.0))

    melody.append(measure_1)
    melody.append(measure_2)
    score.append(melody)
    return score


def _build_harmony_score() -> stream.Score:
    score = stream.Score(id="HarmonyScore")
    part = stream.Part(id="Harmony")
    part.partName = "Harmony"
    part.insert(0, key.Key("C"))
    part.insert(0, meter.TimeSignature("4/4"))

    measure_1 = stream.Measure(number=1)
    measure_1.append(chord.Chord(["C4", "E4", "G4"], quarterLength=2.0))
    measure_1.append(chord.Chord(["G3", "B3", "D4"], quarterLength=2.0))

    measure_2 = stream.Measure(number=2)
    measure_2.append(chord.Chord(["F3", "A3", "C4"], quarterLength=2.0))
    measure_2.append(chord.Chord(["G3", "B3", "D4"], quarterLength=2.0))

    part.append(measure_1)
    part.append(measure_2)
    score.append(part)
    return score


def _build_vertical_consonance_score() -> stream.Score:
    score = stream.Score(id="VerticalConsonance")
    part = stream.Part(id="Harmony")
    part.partName = "Piano"
    part.insert(0, key.Key("F"))
    part.insert(0, meter.TimeSignature("4/4"))

    measure_1 = stream.Measure(number=1)
    measure_1.append(note.Note("F3", quarterLength=1.0))
    measure_1.append(note.Note("A-3", quarterLength=1.0))
    measure_1.append(note.Note("C4", quarterLength=1.0))
    measure_1.append(note.Rest(quarterLength=1.0))

    measure_2 = stream.Measure(number=2)
    measure_2.append(chord.Chord(["F3", "A-3", "C4"], quarterLength=4.0))

    part.append(measure_1)
    part.append(measure_2)
    score.append(part)
    return score


def _build_empty_chord_score() -> stream.Score:
    score = stream.Score(id="EmptyChordScore")
    part = stream.Part(id="SparseHarmony")
    part.partName = "SparseHarmony"
    part.insert(0, key.Key("C"))
    part.insert(0, meter.TimeSignature("4/4"))

    measure_1 = stream.Measure(number=1)
    measure_1.append(chord.Chord([], quarterLength=1.0))
    measure_1.append(note.Note("C4", quarterLength=1.0))
    part.append(measure_1)
    score.append(part)
    return score


def _build_repeated_single_pitch_score() -> stream.Score:
    score = stream.Score(id="RepeatedSinglePitch")
    melody = stream.Part(id="Melody")
    melody.partName = "Melody"
    melody.insert(0, key.Key("C"))
    melody.insert(0, meter.TimeSignature("4/4"))

    measure_1 = stream.Measure(number=1)
    for _ in range(4):
        measure_1.append(note.Note("C4", quarterLength=1.0))

    measure_2 = stream.Measure(number=2)
    for _ in range(4):
        measure_2.append(note.Note("C4", quarterLength=1.0))

    melody.append(measure_1)
    melody.append(measure_2)
    score.append(melody)
    return score


def _build_duplicate_piano_staff_score() -> stream.Score:
    score = stream.Score(id="DuplicatePianoStaves")

    upper_staff = stream.Part(id="UpperStaff")
    upper_staff.partName = "Piano"
    upper_staff.insert(0, key.Key("F"))
    upper_staff.insert(0, meter.TimeSignature("4/4"))

    upper_measure_1 = stream.Measure(number=1)
    for pitch_name in ["F4", "A4", "C5", "F5"]:
        upper_measure_1.append(note.Note(pitch_name, quarterLength=1.0))

    upper_measure_2 = stream.Measure(number=2)
    for pitch_name in ["G4", "B-4", "D5", "G5"]:
        upper_measure_2.append(note.Note(pitch_name, quarterLength=1.0))

    upper_staff.append(upper_measure_1)
    upper_staff.append(upper_measure_2)

    lower_staff = stream.Part(id="LowerStaff")
    lower_staff.partName = "Piano"
    lower_staff.insert(0, key.Key("F"))
    lower_staff.insert(0, meter.TimeSignature("4/4"))

    lower_measure_1 = stream.Measure(number=1)
    lower_measure_1.append(chord.Chord(["F2", "C3", "F3"], quarterLength=4.0))

    lower_measure_2 = stream.Measure(number=2)
    lower_measure_2.append(chord.Chord(["G2", "D3", "G3"], quarterLength=4.0))

    lower_staff.append(lower_measure_1)
    lower_staff.append(lower_measure_2)

    score.append(upper_staff)
    score.append(lower_staff)
    return score


def _build_phrase_priority_score() -> stream.Score:
    score = stream.Score(id="PhrasePriority")
    melody = stream.Part(id="Melody")
    melody.partName = "Melody"
    melody.insert(0, key.Key("C"))
    melody.insert(0, meter.TimeSignature("4/4"))

    measure_1 = stream.Measure(number=1)
    for pitch_name in ["C4", "D4", "E4", "G4"]:
        measure_1.append(note.Note(pitch_name, quarterLength=1.0))

    measure_2 = stream.Measure(number=2)
    for pitch_name in ["A4", "G4"]:
        measure_2.append(note.Note(pitch_name, quarterLength=1.0))
    measure_2.append(note.Rest(quarterLength=2.0))

    measure_3 = stream.Measure(number=3)
    for pitch_name in ["D4", "E4", "F#4", "A4"]:
        measure_3.append(note.Note(pitch_name, quarterLength=1.0))

    measure_4 = stream.Measure(number=4)
    for pitch_name in ["B4", "A4"]:
        measure_4.append(note.Note(pitch_name, quarterLength=1.0))
    measure_4.append(note.Rest(quarterLength=2.0))

    melody.append(measure_1)
    melody.append(measure_2)
    melody.append(measure_3)
    melody.append(measure_4)
    score.append(melody)
    return score


def _build_perfect_authentic_cadence_score() -> stream.Score:
    score = stream.Score(id="PerfectAuthenticCadence")

    upper_staff = stream.Part(id="UpperStaff")
    upper_staff.partName = "Piano"
    upper_staff.insert(0, key.Key("C"))
    upper_staff.insert(0, meter.TimeSignature("4/4"))

    upper_measure_1 = stream.Measure(number=1)
    upper_measure_1.append(note.Note("G4", quarterLength=4.0))

    upper_measure_2 = stream.Measure(number=2)
    upper_measure_2.append(note.Note("C5", quarterLength=4.0))

    upper_staff.append(upper_measure_1)
    upper_staff.append(upper_measure_2)

    lower_staff = stream.Part(id="LowerStaff")
    lower_staff.partName = "Piano"
    lower_staff.insert(0, key.Key("C"))
    lower_staff.insert(0, meter.TimeSignature("4/4"))

    lower_measure_1 = stream.Measure(number=1)
    lower_measure_1.append(chord.Chord(["G2", "B2", "D3"], quarterLength=4.0))

    lower_measure_2 = stream.Measure(number=2)
    lower_measure_2.append(chord.Chord(["C3", "E3", "G3"], quarterLength=4.0))

    lower_staff.append(lower_measure_1)
    lower_staff.append(lower_measure_2)

    score.append(upper_staff)
    score.append(lower_staff)
    return score


def _build_secondary_key_pac_score() -> stream.Score:
    score = stream.Score(id="SecondaryKeyPAC")

    upper_staff = stream.Part(id="UpperStaff")
    upper_staff.partName = "Piano"
    upper_staff.insert(0, key.Key("C"))
    upper_staff.insert(0, meter.TimeSignature("4/4"))

    upper_measure_1 = stream.Measure(number=1)
    upper_measure_1.append(note.Note("A4", quarterLength=4.0))

    upper_measure_2 = stream.Measure(number=2)
    upper_measure_2.append(note.Note("G4", quarterLength=4.0))

    upper_staff.append(upper_measure_1)
    upper_staff.append(upper_measure_2)

    lower_staff = stream.Part(id="LowerStaff")
    lower_staff.partName = "Piano"
    lower_staff.insert(0, key.Key("C"))
    lower_staff.insert(0, meter.TimeSignature("4/4"))

    lower_measure_1 = stream.Measure(number=1)
    lower_measure_1.append(chord.Chord(["D3", "F#3", "A3"], quarterLength=4.0))

    lower_measure_2 = stream.Measure(number=2)
    lower_measure_2.append(chord.Chord(["G2", "B2", "D3"], quarterLength=4.0))

    lower_staff.append(lower_measure_1)
    lower_staff.append(lower_measure_2)

    score.append(upper_staff)
    score.append(lower_staff)
    return score


def _build_measure_lowest_bass_pac_score() -> stream.Score:
    score = stream.Score(id="MeasureLowestBassPAC")

    upper_staff = stream.Part(id="UpperStaff")
    upper_staff.partName = "Piano"
    upper_staff.insert(0, key.Key("C"))
    upper_staff.insert(0, meter.TimeSignature("4/4"))

    upper_measure_1 = stream.Measure(number=1)
    upper_measure_1.append(note.Note("A4", quarterLength=4.0))

    upper_measure_2 = stream.Measure(number=2)
    upper_measure_2.append(note.Note("G4", quarterLength=4.0))

    upper_staff.append(upper_measure_1)
    upper_staff.append(upper_measure_2)

    lower_staff = stream.Part(id="LowerStaff")
    lower_staff.partName = "Piano"
    lower_staff.insert(0, key.Key("C"))
    lower_staff.insert(0, meter.TimeSignature("4/4"))

    lower_measure_1 = stream.Measure(number=1)
    lower_measure_1.append(note.Note("D3", quarterLength=1.0))
    lower_measure_1.append(chord.Chord(["F#3", "A3", "D4"], quarterLength=3.0))

    lower_measure_2 = stream.Measure(number=2)
    lower_measure_2.append(chord.Chord(["G2", "B2", "D3"], quarterLength=4.0))

    lower_staff.append(lower_measure_1)
    lower_staff.append(lower_measure_2)

    score.append(upper_staff)
    score.append(lower_staff)
    return score


def _build_restored_window_pac_score() -> stream.Score:
    score = stream.Score(id="RestoredWindowPAC")

    upper_staff = stream.Part(id="UpperStaff")
    upper_staff.partName = "Piano"
    upper_staff.insert(0, key.Key("C"))
    upper_staff.insert(0, meter.TimeSignature("4/4"))

    upper_measure_1 = stream.Measure(number=1)
    upper_measure_1.append(note.Note("A4", quarterLength=4.0))

    upper_measure_2 = stream.Measure(number=2)
    upper_measure_2.append(note.Note("D5", quarterLength=2.0))
    upper_measure_2.append(note.Note("G4", quarterLength=2.0))

    upper_measure_3 = stream.Measure(number=3)
    upper_measure_3.append(chord.Chord(["B4", "D5", "G5"], quarterLength=2.0))
    upper_measure_3.append(note.Rest(quarterLength=2.0))

    upper_staff.append(upper_measure_1)
    upper_staff.append(upper_measure_2)
    upper_staff.append(upper_measure_3)

    lower_staff = stream.Part(id="LowerStaff")
    lower_staff.partName = "Piano"
    lower_staff.insert(0, key.Key("C"))
    lower_staff.insert(0, meter.TimeSignature("4/4"))

    lower_measure_1 = stream.Measure(number=1)
    lower_measure_1.append(chord.Chord(["D3", "F#3", "A3"], quarterLength=4.0))

    lower_measure_2 = stream.Measure(number=2)
    lower_measure_2.append(note.Note("G2", quarterLength=4.0))

    lower_measure_3 = stream.Measure(number=3)
    lower_measure_3.append(chord.Chord(["G2", "B2", "D3"], quarterLength=4.0))

    lower_staff.append(lower_measure_1)
    lower_staff.append(lower_measure_2)
    lower_staff.append(lower_measure_3)

    score.append(upper_staff)
    score.append(lower_staff)
    return score


def _build_arpeggiated_tonic_window_pac_score() -> stream.Score:
    score = stream.Score(id="ArpeggiatedTonicWindowPAC")

    upper_staff = stream.Part(id="UpperStaff")
    upper_staff.partName = "Piano"
    upper_staff.insert(0, key.Key("G"))
    upper_staff.insert(0, meter.TimeSignature("4/4"))

    upper_measure_1 = stream.Measure(number=1)
    upper_measure_1.append(note.Note("D5", quarterLength=4.0))

    upper_measure_2 = stream.Measure(number=2)
    upper_measure_2.append(note.Note("G5", quarterLength=4.0))

    upper_staff.append(upper_measure_1)
    upper_staff.append(upper_measure_2)

    lower_staff = stream.Part(id="LowerStaff")
    lower_staff.partName = "Piano"
    lower_staff.insert(0, key.Key("G"))
    lower_staff.insert(0, meter.TimeSignature("4/4"))

    lower_measure_1 = stream.Measure(number=1)
    lower_measure_1.append(chord.Chord(["D3", "F#3", "A3", "C4"], quarterLength=4.0))

    lower_measure_2 = stream.Measure(number=2)
    lower_measure_2.append(chord.Chord(["G2", "B2"], quarterLength=0.5))
    lower_measure_2.append(chord.Chord(["B2", "D3"], quarterLength=0.5))
    lower_measure_2.append(chord.Chord(["D3", "G3"], quarterLength=0.5))
    lower_measure_2.append(chord.Chord(["G2", "B2"], quarterLength=0.5))
    lower_measure_2.append(note.Rest(quarterLength=2.0))

    lower_staff.append(lower_measure_1)
    lower_staff.append(lower_measure_2)

    score.append(upper_staff)
    score.append(lower_staff)
    return score


def _build_misaligned_measure_number_score() -> stream.Score:
    score = stream.Score(id="MisalignedMeasureNumbers")

    upper_staff = stream.Part(id="UpperStaff")
    upper_staff.partName = "Piano"
    upper_staff.insert(0, key.Key("C"))
    upper_staff.insert(0, meter.TimeSignature("2/4"))
    for number, pitch_name in [(1, "C5"), (2, "D5"), (3, "E5")]:
        measure_item = stream.Measure(number=number)
        measure_item.append(note.Note(pitch_name, quarterLength=2.0))
        upper_staff.append(measure_item)

    lower_staff = stream.Part(id="LowerStaff")
    lower_staff.partName = "Piano"
    lower_staff.insert(0, key.Key("C"))
    lower_staff.insert(0, meter.TimeSignature("2/4"))
    lower_measure_1 = stream.Measure(number=1)
    long_voice = stream.Voice()
    long_voice.insert(0.0, note.Note("C3", quarterLength=0.5))
    long_voice.insert(2.0, note.Note("D3", quarterLength=0.5))
    long_voice.insert(4.0, note.Note("E3", quarterLength=0.5))
    lower_measure_1.insert(0.0, long_voice)
    lower_staff.append(lower_measure_1)

    score.append(upper_staff)
    score.append(lower_staff)
    return score


class SymbolicAnalysisTestCase(unittest.TestCase):
    def test_exact_recurrence_score_extracts_pitch_statistics(self) -> None:
        result = analyze_symbolic_score(
            _build_exact_recurrence_score(),
            config=SymbolicAnalysisConfig(theme_window_notes=4, max_recurrence_results=8),
        )

        self.assertEqual(result["global_key"], "C大调")
        self.assertEqual(result["total_notes"], 8)
        self.assertEqual(result["unique_pitch_classes"], 4)
        self.assertEqual(len(result["pitch_class_histogram"]), 4)
        self.assertIn("measure_pitch_summary", result)
        self.assertFalse(result["interval_class_histogram"].empty)
        self.assertTrue((result["pitch_class_histogram"]["count"] == 2).all())

    def test_theme_recurrence_detects_exact_and_transposed_matches(self) -> None:
        exact_result = analyze_symbolic_score(
            _build_exact_recurrence_score(),
            config=SymbolicAnalysisConfig(theme_window_notes=4, max_recurrence_results=8),
        )
        transposed_result = analyze_symbolic_score(
            _build_transposed_recurrence_score(),
            config=SymbolicAnalysisConfig(theme_window_notes=4, max_recurrence_results=8),
        )

        self.assertIn("精确再现", set(exact_result["theme_matches"]["relation_type"]))
        self.assertIn("移调再现", set(transposed_result["theme_matches"]["relation_type"]))

    def test_harmony_table_includes_roman_numeral_candidates(self) -> None:
        result = analyze_symbolic_score(_build_harmony_score())
        roman_candidates = set(result["harmony_table"]["roman_numeral"].astype(str))

        self.assertIn("I", roman_candidates)
        self.assertIn("V", roman_candidates)
        self.assertIn("IV", roman_candidates)

    def test_vertical_consonance_ignores_broken_chords_and_keeps_simultaneous_triad_intervals(self) -> None:
        result = analyze_symbolic_score(_build_vertical_consonance_score())

        vertical_table = result["vertical_interval_table"]
        self.assertFalse(vertical_table.empty)
        self.assertEqual(set(vertical_table["measure_number"].astype(int)), {2})
        interval_names = vertical_table.loc[vertical_table["measure_number"].astype(int) == 2, "interval_name"].astype(str).tolist()
        self.assertCountEqual(interval_names, ["小三度", "大三度", "纯五度"])

        summary = result["measure_consonance_summary"]
        self.assertEqual(set(summary["measure_number"].astype(int)), {2})
        measure_2 = summary.iloc[0]
        self.assertEqual(int(measure_2["vertical_interval_count"]), 3)
        self.assertAlmostEqual(float(measure_2["mean_consonance_rank"]), (6 + 5 + 3) / 3, places=3)

    def test_empty_chord_events_are_skipped_safely(self) -> None:
        result = analyze_symbolic_score(_build_empty_chord_score())

        self.assertEqual(result["total_notes"], 1)
        self.assertEqual(result["unique_pitch_classes"], 1)
        self.assertFalse(result["note_table"].empty)

    def test_export_symbolic_score_file_builds_musicxml_bytes(self) -> None:
        data, filename, mime = export_symbolic_score_file(_build_harmony_score(), fmt="musicxml")

        self.assertTrue(data.startswith(b"<?xml") or b"<score-partwise" in data[:200])
        self.assertTrue(filename.endswith(".musicxml"))
        self.assertEqual(mime, "application/vnd.recordare.musicxml+xml")

    def test_repeated_single_pitch_windows_are_not_treated_as_theme_matches(self) -> None:
        result = analyze_symbolic_score(
            _build_repeated_single_pitch_score(),
            config=SymbolicAnalysisConfig(theme_window_notes=6, max_recurrence_results=12),
        )

        self.assertTrue(result["theme_matches"].empty)

    def test_duplicate_piano_staff_names_do_not_overwrite_melodic_sequences(self) -> None:
        result = analyze_symbolic_score(
            _build_duplicate_piano_staff_score(),
            config=SymbolicAnalysisConfig(theme_window_notes=4, max_recurrence_results=12),
        )

        part_names = set(result["note_table"]["part_name"].astype(str))
        self.assertIn("Piano", part_names)
        self.assertIn("Piano [2]", part_names)
        self.assertGreaterEqual(len(result["part_summary"]), 2)
        self.assertFalse(result["theme_matches"].empty)
        self.assertEqual(set(result["theme_matches"]["source_part"].astype(str)), {"Piano"})
        self.assertEqual(set(result["theme_matches"]["match_part"].astype(str)), {"Piano"})

    def test_theme_matches_prioritize_phrase_start_sequence(self) -> None:
        result = analyze_symbolic_score(
            _build_phrase_priority_score(),
            config=SymbolicAnalysisConfig(theme_window_notes=6, max_recurrence_results=12),
        )

        top_match = result["theme_matches"].iloc[0]
        self.assertEqual(str(top_match["relation_type"]), "移调再现")
        self.assertEqual(int(top_match["source_measure"]), 1)
        self.assertAlmostEqual(float(top_match["source_beat"]), 1.0)
        self.assertEqual(int(top_match["match_measure"]), 3)
        self.assertAlmostEqual(float(top_match["match_beat"]), 1.0)

    def test_perfect_authentic_cadence_candidates_detect_v_to_i_with_tonic_melody(self) -> None:
        result = analyze_symbolic_score(_build_perfect_authentic_cadence_score())

        self.assertFalse(result["cadence_candidates"].empty)
        pac_candidates = result["cadence_candidates"].loc[
            result["cadence_candidates"]["cadence_type"].astype(str) == "完满终止候选"
        ].reset_index(drop=True)
        self.assertFalse(pac_candidates.empty)
        top_candidate = pac_candidates.iloc[0]
        self.assertEqual(str(top_candidate["cadence_type"]), "完满终止候选")
        self.assertEqual(int(top_candidate["measure_number"]), 2)
        self.assertEqual(str(top_candidate["previous_roman_numeral"]), "V")
        self.assertEqual(str(top_candidate["current_roman_numeral"]), "I")
        self.assertEqual(int(top_candidate["previous_bass_scale_degree"]), 5)
        self.assertEqual(int(top_candidate["current_bass_scale_degree"]), 1)
        self.assertEqual(int(top_candidate["melody_scale_degree"]), 1)

    def test_perfect_authentic_cadence_candidates_can_use_local_key(self) -> None:
        result = analyze_symbolic_score(_build_secondary_key_pac_score())

        pac_candidates = result["cadence_candidates"].loc[
            result["cadence_candidates"]["cadence_type"].astype(str) == "完满终止候选"
        ].reset_index(drop=True)
        self.assertFalse(pac_candidates.empty)
        top_candidate = pac_candidates.iloc[0]
        self.assertEqual(str(top_candidate["cadence_key"]), "G大调")
        self.assertEqual(str(top_candidate["global_key"]), "C大调")
        self.assertEqual(str(top_candidate["previous_roman_numeral"]), "V")
        self.assertEqual(str(top_candidate["current_roman_numeral"]), "I")
        self.assertEqual(int(top_candidate["melody_scale_degree"]), 1)

    def test_perfect_authentic_cadence_candidates_use_measure_lowest_bass(self) -> None:
        result = analyze_symbolic_score(_build_measure_lowest_bass_pac_score())

        pac_candidates = result["cadence_candidates"].loc[
            result["cadence_candidates"]["cadence_type"].astype(str) == "完满终止候选"
        ].reset_index(drop=True)
        self.assertFalse(pac_candidates.empty)
        top_candidate = pac_candidates.iloc[0]
        self.assertEqual(str(top_candidate["cadence_key"]), "G大调")
        self.assertEqual(str(top_candidate["previous_bass_pitch_name"]), "D3")
        self.assertEqual(int(top_candidate["previous_bass_scale_degree"]), 5)
        self.assertEqual(str(top_candidate["current_bass_pitch_name"]), "G2")
        self.assertEqual(int(top_candidate["current_bass_scale_degree"]), 1)
        self.assertEqual(str(top_candidate["melody_skeleton_class"]), "主音骨架（1）")

    def test_perfect_authentic_cadence_candidates_delay_restored_window_until_literal_tonic(self) -> None:
        result = analyze_symbolic_score(_build_restored_window_pac_score())

        pac_candidates = result["cadence_candidates"].loc[
            result["cadence_candidates"]["cadence_type"].astype(str) == "完满终止候选"
        ].reset_index(drop=True)
        self.assertFalse(pac_candidates.empty)
        top_candidate = pac_candidates.iloc[0]
        self.assertEqual(int(top_candidate["measure_number"]), 3)
        self.assertEqual(str(top_candidate["cadence_key"]), "G大调")
        self.assertEqual(str(top_candidate["previous_roman_numeral"]), "V")
        self.assertEqual(str(top_candidate["current_roman_numeral"]), "I")

    def test_perfect_authentic_cadence_candidates_keep_arpeggiated_tonic_window(self) -> None:
        result = analyze_symbolic_score(_build_arpeggiated_tonic_window_pac_score())

        pac_candidates = result["cadence_candidates"].loc[
            result["cadence_candidates"]["cadence_type"].astype(str) == "完满终止候选"
        ].reset_index(drop=True)
        self.assertFalse(pac_candidates.empty)
        top_candidate = pac_candidates.iloc[0]
        self.assertEqual(int(top_candidate["measure_number"]), 2)
        self.assertEqual(str(top_candidate["cadence_key"]), "G大调")
        self.assertEqual(str(top_candidate["previous_roman_numeral"]), "V7")
        self.assertEqual(str(top_candidate["current_roman_numeral"]), "I")
        self.assertGreaterEqual(float(top_candidate["closure_score"]), 0.70)
        self.assertEqual(str(top_candidate["melody_pitch_name"]), "G5")
        self.assertEqual(int(top_candidate["melody_scale_degree"]), 1)

    def test_half_cadence_candidates_detect_dominant_arrival_without_tonic_resolution(self) -> None:
        score = stream.Score(id="HalfCadence")

        upper_staff = stream.Part(id="UpperStaff")
        upper_staff.partName = "Piano"
        upper_staff.insert(0, key.Key("C"))
        upper_staff.insert(0, meter.TimeSignature("4/4"))
        measure_1 = stream.Measure(number=1)
        measure_1.append(note.Note("B4", quarterLength=4.0))
        measure_2 = stream.Measure(number=2)
        measure_2.append(note.Note("G4", quarterLength=4.0))
        upper_staff.append(measure_1)
        upper_staff.append(measure_2)

        lower_staff = stream.Part(id="LowerStaff")
        lower_staff.partName = "Piano"
        lower_staff.insert(0, key.Key("C"))
        lower_staff.insert(0, meter.TimeSignature("4/4"))
        lower_measure_1 = stream.Measure(number=1)
        lower_measure_1.append(chord.Chord(["D3", "F3", "A3"], quarterLength=4.0))
        lower_measure_2 = stream.Measure(number=2)
        lower_measure_2.append(chord.Chord(["G2", "B2", "D3"], quarterLength=4.0))
        lower_staff.append(lower_measure_1)
        lower_staff.append(lower_measure_2)

        score.append(upper_staff)
        score.append(lower_staff)

        result = analyze_symbolic_score(score)
        hc_candidates = result["cadence_candidates"].loc[
            result["cadence_candidates"]["cadence_type"].astype(str) == "半终止候选"
        ].reset_index(drop=True)
        self.assertFalse(hc_candidates.empty)
        top_candidate = hc_candidates.iloc[0]
        self.assertEqual(int(top_candidate["measure_number"]), 2)
        self.assertEqual(str(top_candidate["cadence_key"]), "C大调")
        self.assertEqual(str(top_candidate["current_roman_numeral"]), "V")
        self.assertEqual(int(top_candidate["melody_scale_degree"]), 5)

    def test_build_note_table_remaps_measure_numbers_from_reference_timeline(self) -> None:
        result = analyze_symbolic_score(_build_misaligned_measure_number_score())

        lower_rows = result["note_table"].loc[result["note_table"]["part_name"].astype(str) == "Piano [2]"].copy()
        self.assertEqual(lower_rows["measure_number"].astype(int).tolist(), [1, 2, 3])
        self.assertEqual(lower_rows["pitch_name"].astype(str).tolist(), ["C3", "D3", "E3"])


if __name__ == "__main__":
    unittest.main()
