from __future__ import annotations

import json
from functools import lru_cache

from paper2_final.paths import CONFIG_DIR


@lru_cache(maxsize=None)
def load_config(name: str) -> dict:
    return json.loads((CONFIG_DIR / f"{name}.json").read_text(encoding="utf-8"))


def bands() -> list[str]:
    return list(load_config("model")["bands"])


def input_name(stage: str, variant: str) -> str:
    aliases = {
        "NO_DYN_ALL": "NO_DYN",
        "NO_SETTLE_ALL": "NO_SETTLE",
        "NO_TEMP_ALL": "NO_TEMP",
    }
    return f"{stage}_{aliases.get(variant, variant)}.npz"
