from __future__ import annotations

import unittest
from unittest import mock
from urllib.error import URLError

from spectral_tool.when_in_rome import _entry_from_tree_path, download_when_in_rome_score, list_when_in_rome_scores


class WhenInRomeCatalogTestCase(unittest.TestCase):
    def test_entry_from_tree_path_extracts_catalog_fields(self) -> None:
        entry = _entry_from_tree_path(
            "Corpus/Keyboard_Other/Bach,_Johann_Sebastian/The_Well-Tempered_Clavier_I/01/score.mxl"
        )

        self.assertEqual(entry["category"], "Keyboard_Other")
        self.assertEqual(entry["category_label"], "Keyboard Other")
        self.assertEqual(entry["composer"], "Bach,_Johann_Sebastian")
        self.assertEqual(entry["composer_label"], "Bach, Johann Sebastian")
        self.assertEqual(entry["collection"], "The_Well-Tempered_Clavier_I")
        self.assertEqual(entry["collection_label"], "The Well-Tempered Clavier I")
        self.assertEqual(entry["item"], "01")
        self.assertIn("Bach, Johann Sebastian", entry["display_name"])
        self.assertIn("raw.githubusercontent.com", entry["raw_url"])
        self.assertTrue(entry["raw_url"].endswith("score.mxl"))

    def test_entry_from_tree_path_handles_deeper_item_paths(self) -> None:
        entry = _entry_from_tree_path(
            "Corpus/OpenScore-LiederCorpus/Brahms,_Johannes/6_Songs,_Op.3/1_Liebestreu/score.mxl"
        )

        self.assertEqual(entry["category_label"], "OpenScore-LiederCorpus")
        self.assertEqual(entry["composer_label"], "Brahms, Johannes")
        self.assertEqual(entry["collection_label"], "6 Songs, Op.3")
        self.assertEqual(entry["item"], "1 Liebestreu")
        self.assertIn("1 Liebestreu", entry["display_name"])

    @mock.patch("spectral_tool.when_in_rome.urllib.request.urlopen", side_effect=URLError("reset"))
    def test_catalog_network_failure_raises_friendly_runtime_error(self, _: mock.Mock) -> None:
        with self.assertRaises(RuntimeError) as context:
            list_when_in_rome_scores()
        self.assertIn("When-in-Rome", str(context.exception))

    @mock.patch("spectral_tool.when_in_rome.urllib.request.urlopen", side_effect=URLError("reset"))
    def test_download_network_failure_raises_friendly_runtime_error(self, _: mock.Mock) -> None:
        with self.assertRaises(RuntimeError) as context:
            download_when_in_rome_score("Corpus/Keyboard_Other/Bach,_Johann_Sebastian/The_Well-Tempered_Clavier_I/01/score.mxl")
        self.assertIn("When-in-Rome", str(context.exception))


if __name__ == "__main__":
    unittest.main()
