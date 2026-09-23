"""Paths and fixed protocol for the full-spectrum data-only experiment."""

import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parents[1]
DATA_ROOT = Path(os.environ.get("PAPER2_DATA_ROOT", PROJECT_ROOT / "data"))
RESULT_DIR = (
    Path(os.environ.get("PAPER2_OUTPUT_DIR", PROJECT_ROOT / "output"))
    / "pure_blackbox_full_spectrum"
)
INPUT_DIR = Path(os.environ.get("PAPER2_INPUT_DIR", DATA_ROOT / "inputs"))
LOCK_FILE = DATA_ROOT / "DATASET_LOCK.json"
BASELINE_FILE = DATA_ROOT / "DATASET_BASELINE.json"
AUTHORITATIVE_DATASET_ROOT = Path(os.environ.get("PAPER2_AUTHORITATIVE_DATASET_ROOT", DATA_ROOT))

SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)
MODELS = (
    "BASGT-DATA-K3",
    "Ridge-DATA",
    "LSTM-DATA",
    "DLinear-DATA",
    "iTransformer-DATA",
    "TimeXer-DATA",
)
NEURAL_MODELS = tuple(name for name in MODELS if name != "Ridge-DATA")
BANDS = ("lt42", "w42_70", "w70_120", "w120_200", "gt200")
SEGMENT_COUNTS = (204, 102, 51, 34, 17)

TRAINING = {
    "learning_rates": [0.001, 0.0003],
    "basgt_learning_rate": 0.001,
    "max_epochs": 300,
    "patience": 40,
    "min_delta": 0.0001,
    "weight_decay": 0.001,
    "batch_size": 2,
    "gradient_clip": 1.0,
}

BASE_FEATURE_NAMES = (
    "hybrid_prior_mm",
    "hybrid_prior_available",
    "physics_mm",
    "physics_available",
    "dynamic_mm",
    "dynamic_available",
    "speed_kmh",
    "monitoring_band_value_mm",
    "monitoring_available",
    "monitoring_chord_profile_mm",
    "monitoring_spatial_slope_mm_per_m",
    "nearest_monitoring_sensor_distance_fraction",
    "monitoring_channel_coverage",
    "wsw_midnight_temperature_c",
    "general_temperature_c",
    "structure_temperature_c",
    "wsw_temperature_available",
    "general_temperature_available",
    "structure_temperature_available",
    "midwave_physics_gate_weight",
    "normalized_mileage",
)
RAW_SENSOR_NAMES = (
    "JKBD36",
    "JKBD34",
    "JKBD32",
    "JKBD30",
    "JKBD28",
    "JKBD26",
    "JKBD24",
    "JKBD22",
    "JKBD20",
    "JKBD18",
    "JKBD16",
    "JKBD14",
    "JKBD12",
    "JKBD10",
    "JKBD08",
    "JKBD06",
    "JKBD04",
    "JKBD02",
    "ND-05-01",
    "ND-06-01",
    "ND-07-01",
    "ND-08-01",
    "ND-09-01",
    "ND-10-01",
    "ND-11-01",
    "SZY-02-02",
    "SZY-03-02",
    "SZY-04-01",
    "SZY-12-01",
    "SZY-13-02",
    "SZY-14-02",
)
FEATURE_NAMES = (
    *BASE_FEATURE_NAMES,
    *(f"raw__{name}" for name in RAW_SENSOR_NAMES),
    *(f"available__{name}" for name in RAW_SENSOR_NAMES),
    *(f"band_identity__{name}" for name in BANDS),
)

# These are white-box or white-box-derived channels and must be zero in NO_WB_ALL.
FORBIDDEN_CHANNEL_INDICES = (0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 19)
