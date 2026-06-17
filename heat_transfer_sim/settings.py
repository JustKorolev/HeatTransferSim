"""Persistent settings helpers for simulation parameters."""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar


SettingsT = TypeVar("SettingsT")


DEFAULT_PARAMETER_FILE = Path(__file__).resolve().parent.parent / "simulation_parameters.json"


MigrationFunction = Callable[[dict[str, Any]], dict[str, Any]]


class ParameterStore(Generic[SettingsT]):
    """Load and save dataclass-backed UI parameters as JSON."""

    def __init__(
        self,
        settings_type: type[SettingsT],
        path: Path = DEFAULT_PARAMETER_FILE,
        migrate: MigrationFunction | None = None,
    ):
        if not is_dataclass(settings_type):
            raise TypeError("settings_type must be a dataclass type")
        self.settings_type = settings_type
        self.path = path
        self.migrate = migrate

    def load(self) -> SettingsT:
        """Load settings from disk, falling back to dataclass defaults."""
        defaults = self.settings_type()
        if not self.path.exists():
            self.save(defaults)
            return defaults

        with self.path.open("r", encoding="utf-8") as file:
            raw_settings = json.load(file)
        if self.migrate is not None:
            raw_settings = self.migrate(raw_settings)

        valid_field_names = {field.name for field in fields(self.settings_type)}
        merged_settings = asdict(defaults)
        merged_settings.update(
            {
                key: value
                for key, value in raw_settings.items()
                if key in valid_field_names
            }
        )
        settings = self.settings_type(**merged_settings)
        self.save(settings)
        return settings

    def save(self, settings: SettingsT) -> None:
        """Write settings to disk as pretty JSON."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(asdict(settings), file, indent=2, sort_keys=True)
            file.write("\n")
