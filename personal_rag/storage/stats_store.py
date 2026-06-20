from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from personal_rag.storage.atomic import atomic_write_json


class StatsStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Stats file must contain an object: {self.path}")
        return value

    def save(self, stats: dict[str, Any]) -> None:
        atomic_write_json(self.path, stats)

    def reset(self) -> None:
        self.path.unlink(missing_ok=True)

