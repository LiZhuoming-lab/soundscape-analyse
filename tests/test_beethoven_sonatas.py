from __future__ import annotations

import unittest
from unittest import mock
from urllib.error import URLError

from spectral_tool.beethoven_sonatas import (
    _entry_from_tree_path,
    download_beethoven_sonata_score,
    list_beethoven_sonata_scores,
)


class BeethovenSonataCatalogTestCase(unittest.TestCase):
    def test_entry_from_tree_path_extracts_sonata_and_movement(self) -> None:
        entry = _entry_from_tree_path("kern/sonata14-1.krn")

        self.assertEqual(entry["sonata_number"], 14)
        self.assertEqual(entry["movement_number"], 1)
        self.assertEqual(entry["score_name"], "sonata14-1.krn")
        self.assertEqual(entry["extension"], ".krn")
        self.assertIn("Beethoven Piano Sonata No.14 / Movement 1", entry["display_name"])
        self.assertIn("raw.githubusercontent.com", entry["raw_url"])

    def test_entry_from_tree_path_handles_nonmatching_paths(self) -> None:
        entry = _entry_from_tree_path("reference-edition/sonata14.pdf")

        self.assertEqual(entry["sonata_number"], 0)
        self.assertEqual(entry["movement_number"], 0)
        self.assertEqual(entry["score_name"], "sonata14.pdf")
        self.assertEqual(entry["display_name"], "sonata14.pdf")

    @mock.patch("spectral_tool.beethoven_sonatas.urllib.request.urlopen", side_effect=URLError("reset"))
    def test_catalog_network_failure_raises_friendly_runtime_error(self, _: mock.Mock) -> None:
        with self.assertRaises(RuntimeError) as context:
            list_beethoven_sonata_scores()
        self.assertIn("Beethoven piano sonatas", str(context.exception))

    @mock.patch("spectral_tool.beethoven_sonatas.urllib.request.urlopen", side_effect=URLError("reset"))
    def test_download_network_failure_raises_friendly_runtime_error(self, _: mock.Mock) -> None:
        with self.assertRaises(RuntimeError) as context:
            download_beethoven_sonata_score("kern/sonata14-1.krn")
        self.assertIn("Beethoven piano sonatas", str(context.exception))


if __name__ == "__main__":
    unittest.main()
