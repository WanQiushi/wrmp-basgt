"""Train graph-depth K=2 and K=3 on the locked four-band setup."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from paper2_final.paths import INPUT_DIR

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / "v3_graph_five_band_7m_center_20260919"
INPUT = INPUT_DIR
sys.path.insert(0, str(ROOT.parent / "v3_graph_topology_ablation_20260917"))
import run_factorial as core
from model import DeepFourBandGT

COUNTS = (204, 102, 51, 34)
DEPTHS = {"graph_k2": 2, "graph_k3": 3}
core.ROOT = ROOT
for variant in DEPTHS:
    core.VARIANTS[variant] = (COUNTS, "adjacent")
active_boundaries = None


def load_stage(stage, variant, device):
    global active_boundaries
    values = dict(np.load(INPUT / f"{stage}_FULL.npz"))
    target = np.load(INPUT / "target.npz")["target"][..., :4]
    prior = values["prior"][..., :4]
    fit = values["fit"].astype(int)
    scale = np.maximum(np.sqrt(np.mean((target[fit] - prior[fit]) ** 2, axis=(0, 1))), 1e-3).astype(
        np.float32
    )
    tensors = {
        "features": torch.tensor(
            values["features"][:, :, :4, :-5].copy(), device=device, dtype=torch.float32
        ),
        "prior": torch.tensor(prior, device=device, dtype=torch.float32),
        "target": torch.tensor(target, device=device, dtype=torch.float32),
        "scale": torch.tensor(scale, device=device),
    }
    active_boundaries = [
        np.load(PARENT / f"{stage}_b{band}_k{count}_segments.npz")["bounds_m"]
        for band, count in enumerate((204, 102, 51, 34, 17))
    ]
    from paper2_final.models.graph_transformer import band_layout

    base = band_layout((17,) * 5)
    physical_ids = torch.as_tensor(base["physical_ids"].copy(), dtype=torch.long, device=device)
    maximum = max(COUNTS)
    point_ids = np.zeros((4, 1433), dtype=np.int64)
    edge_type = np.zeros((4, maximum, maximum), dtype=np.int64)
    edge_weight = np.zeros((4, maximum, maximum), dtype=np.float32)
    for band, bounds in enumerate(active_boundaries[:4]):
        count = COUNTS[band]
        point_ids[band] = np.searchsorted(bounds[1:-1], np.arange(318206, 319639), side="right")
        starts, ends = bounds[:-1], bounds[1:]
        zones = np.where(ends <= 318376, 0, np.where(starts >= 319468, 2, 1))
        for node in range(count - 1):
            crossing = zones[node] != zones[node + 1]
            edge_type[band, node, node + 1] = edge_type[band, node + 1, node] = 2 if crossing else 1
            edge_weight[band, node, node + 1] = edge_weight[band, node + 1, node] = (
                0.25 if crossing else 1.0
            )
        np.fill_diagonal(edge_weight[band], 1.0)
    graph = (
        physical_ids,
        torch.as_tensor(point_ids, dtype=torch.long, device=device),
        torch.as_tensor(edge_type, dtype=torch.long, device=device),
        torch.as_tensor(edge_weight, dtype=torch.float32, device=device),
    )
    return values, tensors, graph


def build_model(variant, feature_dim, device):
    return DeepFourBandGT(feature_dim, active_boundaries, graph_depth=DEPTHS[variant]).to(device)


core.load_stage = load_stage
core.build_model = build_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(DEPTHS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    core.train(args.variant, args.seed, args.device)
