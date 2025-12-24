"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return a dict."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
