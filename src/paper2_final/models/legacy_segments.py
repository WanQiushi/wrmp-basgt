"""Nested structural and wavelength-specific segments for Wufengshan.

The evaluation interval is 318206--319638 m.  The bridge drawing gives two
2 m end pieces, 12 x 14 m on each side of the towers, and 78 x 14 m in the
main span.  The resulting 102 atomic 14 m cells support an exact hierarchy:

* six atoms -> one 84 m physical stiffness segment (17 total);
* three atoms -> one 70--120 m observation segment (34 total);
* two atoms -> one 42--70 m observation segment (51 total);
* one atom  -> one <42 m observation segment (102 total).

The 2 m end pieces are merged into the first and last segment at every level.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

CORE_START_M = 318_206.0
INNER_START_M = 318_208.0
LEFT_TOWER_M = 318_376.0
MIDSPAN_M = 318_922.0
RIGHT_TOWER_M = 319_468.0
INNER_END_M = 319_636.0
CORE_END_M = 319_638.0
ATOMIC_LENGTH_M = 14.0
N_ATOMIC_CELLS = 102


@dataclass(frozen=True)
class Segment:
    segment_id: str
    level: str
    index: int
    start_m: float
    end_m: float
    region: str
    atomic_start: int
    atomic_end: int

    @property
    def length_m(self) -> float:
        return float(self.end_m - self.start_m)

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "length_m": self.length_m}


def _region(start_m: float, end_m: float) -> str:
    if end_m <= LEFT_TOWER_M:
        return "left_side"
    if start_m >= RIGHT_TOWER_M:
        return "right_side"
    if start_m <= MIDSPAN_M <= end_m:
        return "main_midspan"
    return "main_span"


def build_nested_segments(group_atoms: int, level: str, prefix: str) -> tuple[Segment, ...]:
    if group_atoms <= 0 or N_ATOMIC_CELLS % int(group_atoms) != 0:
        raise ValueError("group_atoms must divide the 102 structural atomic cells")
    count = N_ATOMIC_CELLS // int(group_atoms)
    segments: list[Segment] = []
    for index in range(count):
        atom_start = index * int(group_atoms)
        atom_end = (index + 1) * int(group_atoms)
        start = INNER_START_M + atom_start * ATOMIC_LENGTH_M
        end = INNER_START_M + atom_end * ATOMIC_LENGTH_M
        if index == 0:
            start = CORE_START_M
        if index == count - 1:
            end = CORE_END_M
        segments.append(
            Segment(
                segment_id=f"{prefix}{index + 1:03d}",
                level=level,
                index=index,
                start_m=float(start),
                end_m=float(end),
                region=_region(float(start), float(end)),
                atomic_start=atom_start,
                atomic_end=atom_end,
            )
        )
    _validate_tiling(tuple(segments))
    return tuple(segments)


def physical_stiffness_segments() -> tuple[Segment, ...]:
    return build_nested_segments(6, "physical_stiffness", "K")


def observation_segments_by_band() -> dict[str, tuple[Segment, ...]]:
    return {
        "lt42": build_nested_segments(1, "observation_lt42", "B1_"),
        "w42_70": build_nested_segments(2, "observation_w42_70", "B2_"),
        "w70_120": build_nested_segments(3, "observation_w70_120", "B3_"),
    }


def _validate_tiling(segments: tuple[Segment, ...]) -> None:
    if not segments:
        raise ValueError("segment collection is empty")
    if not np.isclose(segments[0].start_m, CORE_START_M):
        raise ValueError("segments do not start at the core boundary")
    if not np.isclose(segments[-1].end_m, CORE_END_M):
        raise ValueError("segments do not end at the core boundary")
    for left, right in zip(segments[:-1], segments[1:]):
        if not np.isclose(left.end_m, right.start_m):
            raise ValueError("segment tiling has a gap or overlap")


def segment_index(mileage_m: np.ndarray, segments: tuple[Segment, ...]) -> np.ndarray:
    x = np.asarray(mileage_m, float)
    boundaries = np.asarray([item.end_m for item in segments[:-1]], float)
    indices = np.searchsorted(boundaries, x, side="right")
    outside = (x < segments[0].start_m) | (x > segments[-1].end_m)
    indices = indices.astype(int)
    indices[outside] = -1
    return indices


def overlap_matrix(physical: tuple[Segment, ...], observations: tuple[Segment, ...]) -> np.ndarray:
    """Return observation-length-normalized overlap with physical segments."""
    matrix = np.zeros((len(observations), len(physical)), dtype=float)
    for row, observation in enumerate(observations):
        for column, item in enumerate(physical):
            overlap = max(
                0.0,
                min(observation.end_m, item.end_m) - max(observation.start_m, item.start_m),
            )
            matrix[row, column] = overlap / observation.length_m
    if not np.allclose(matrix.sum(axis=1), 1.0):
        raise RuntimeError("observation-to-physical overlap rows must sum to one")
    return matrix


def expand_physical_values(
    mileage_m: np.ndarray,
    values: np.ndarray,
    segments: tuple[Segment, ...] | None = None,
) -> np.ndarray:
    segments = physical_stiffness_segments() if segments is None else segments
    values = np.asarray(values, float)
    if len(values) != len(segments):
        raise ValueError("one physical value is required per stiffness segment")
    indices = segment_index(np.asarray(mileage_m, float), segments)
    output = np.empty(len(indices), dtype=float)
    inside = indices >= 0
    output[inside] = values[indices[inside]]
    output[~inside & (np.asarray(mileage_m) < CORE_START_M)] = values[0]
    output[~inside & (np.asarray(mileage_m) > CORE_END_M)] = values[-1]
    return output


def region_parameter_map(model: str) -> np.ndarray:
    """Map 17 physical segments to a model's free parameter indices."""
    if model == "uniform_k":
        return np.zeros(17, dtype=int)
    if model == "three_region_k":
        return np.asarray([0, 0, *([1] * 13), 2, 2], dtype=int)
    if model == "segment17_k":
        return np.arange(17, dtype=int)
    if model == "symmetric9_k":
        return np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    raise ValueError(f"unknown stiffness parameterization: {model}")
