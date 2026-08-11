"""Small JSON-config helpers for fidelity CLIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


def load_config_defaults(argv: Optional[Sequence[str]] = None) -> Tuple[Dict[str, Any], Optional[Path]]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args(argv)
    if pre_args.config is None:
        return {}, None
    with pre_args.config.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a JSON object: {pre_args.config}")
    return data, pre_args.config
