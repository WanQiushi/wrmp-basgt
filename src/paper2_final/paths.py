from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DATA_ROOT = Path(os.environ.get("PAPER2_DATA_ROOT", PROJECT_ROOT / "data"))
INPUT_DIR = Path(os.environ.get("PAPER2_INPUT_DIR", DATA_ROOT / "inputs"))
RUN_DIR = PROJECT_ROOT / "runs"
REPORT_DIR = PROJECT_ROOT / "reports"


def ensure_output_directories() -> None:
    RUN_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
