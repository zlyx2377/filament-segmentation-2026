"""Config loader with optional `_base_` inheritance (deep-merged)."""
from __future__ import annotations

import os
import copy
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Accept both ``_base`` and ``_base_`` as the inheritance key.
    base_key = next((k for k in ("_base", "_base_") if k in raw), None)
    if base_key:
        base_path = os.path.join(os.path.dirname(path), raw.pop(base_key))
        base = load_config(base_path)
        return _deep_merge(base, raw)
    return raw
