"""Tests for persistent parameter storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from heat_transfer_sim.settings import ParameterStore


@dataclass
class ExampleSettings:
    temperature: float = 20.0
    num_points: int = 100


class ParameterStoreTests(unittest.TestCase):
    def test_load_creates_default_settings_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "params.json"
            store = ParameterStore(ExampleSettings, path)
            settings = store.load()
            self.assertEqual(settings, ExampleSettings())
            self.assertTrue(path.exists())

    def test_saved_settings_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "params.json"
            store = ParameterStore(ExampleSettings, path)
            store.save(ExampleSettings(temperature=45.0, num_points=500))
            self.assertEqual(
                store.load(),
                ExampleSettings(temperature=45.0, num_points=500),
            )


if __name__ == "__main__":
    unittest.main()
