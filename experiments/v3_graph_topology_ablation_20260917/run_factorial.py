"""Parameter-matched routed v3-core graph topology factorial on locked V4 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper2_final.config import load_config
from paper2_final.models.graph_transformer import BandMultiResolutionFullGT, multires_graph_tensors
from paper2_final.paths import INPUT_DIR
from paper2_final.reproducibility import set_seed

ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "seg17_adj": ((17, 17, 17, 17, 17), "adjacent"),
    "seg17_full": ((17, 17, 17, 17, 17), "full"),
    "multires_adj": ((34, 17, 9, 5, 3), "adjacent"),
    "multires_full": ((34, 17, 9, 5, 3), "full"),
}


def load_stage(stage: str, variant: str, device: str):
    values = dict(np.load(INPUT_DIR / f"{stage}_FULL.npz"))
    target = np.load(INPUT_DIR / "target.npz")["target"]
    fit = values["fit"].astype(int)
    scale = np.maximum(
        np.sqrt(np.mean((target[fit] - values["prior"][fit]) ** 2, axis=(0, 1))),
        1e-3,
    ).astype(np.float32)
    tensors = {
        "features": torch.tensor(
            values["features"][..., :-5].copy(), device=device, dtype=torch.float32
        ),
        "prior": torch.tensor(values["prior"], device=device, dtype=torch.float32),
        "target": torch.tensor(target, device=device, dtype=torch.float32),
        "scale": torch.tensor(scale, device=device),
    }
    counts, connectivity = VARIANTS[variant]
    graph = multires_graph_tensors(values["segments"], counts, 0.25, device)
    if connectivity == "adjacent":
        positions = torch.arange(graph[3].shape[1], device=device)
        adjacent = (positions[:, None] - positions[None, :]).abs() <= 1
        graph = (*graph[:3], graph[3] * adjacent[None])
    return values, tensors, graph


def build_model(variant: str, feature_dim: int, device: str):
    counts, _ = VARIANTS[variant]
    return BandMultiResolutionFullGT(
        feature_dim,
        band_segment_counts=counts,
        d_model=368,
        heads=8,
        ff_dim=736,
        local_kernel_size=9,
        dropout=0.1,
    ).to(device)


def predict(model, tensors, graph, indices):
    residual = model(tensors["features"][indices], *graph)
    return tensors["prior"][indices] + residual * tensors["scale"]


def evaluate(model, tensors, graph, indices):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(indices), 2):
            outputs.append(predict(model, tensors, graph, indices[start : start + 2]))
    prediction = torch.cat(outputs)
    loss = torch.mean(((prediction - tensors["target"][indices]) / tensors["scale"]) ** 2)
    return float(loss), prediction.cpu().numpy()


def train(variant: str, seed: int, device: str):
    training = load_config("model")["training"]
    output = ROOT / "factorial_runs" / variant / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    selection, tensors, graph = load_stage("selection", variant, device)
    set_seed(seed)
    model = build_model(variant, tensors["features"].shape[-1], device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=training["weight_decay"]
    )
    generator = np.random.default_rng(seed)
    best, selected_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, training["max_epochs"] + 1):
        model.train()
        losses = []
        order = generator.permutation(selection["fit"].astype(int))
        for start in range(0, len(order), training["batch_size"]):
            indices = order[start : start + training["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            prediction = predict(model, tensors, graph, indices)
            loss = torch.mean(((prediction - tensors["target"][indices]) / tensors["scale"]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_loss, _ = evaluate(model, tensors, graph, selection["val"].astype(int))
        history.append(
            {
                "stage": "selection",
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best - training["min_delta"]:
            best, selected_epoch, stale = validation_loss, epoch, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{variant} seed={seed} selection epoch={epoch} "
                f"val={validation_loss:.6f} best={best:.6f} "
                f"selected={selected_epoch}",
                flush=True,
            )
        if stale >= training["patience"]:
            break
    (output / "selection.json").write_text(
        json.dumps(
            {
                "learning_rate": 0.001,
                "epochs": selected_epoch,
                "validation_loss": best,
                "parameters": parameter_count,
            },
            indent=2,
        )
        + "\n"
    )

    del model, optimizer, tensors, graph
    final, tensors, graph = load_stage("final", variant, device)
    set_seed(seed)
    model = build_model(variant, tensors["features"].shape[-1], device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=training["weight_decay"]
    )
    generator = np.random.default_rng(seed)
    for epoch in range(1, selected_epoch + 1):
        model.train()
        losses = []
        order = generator.permutation(final["fit"].astype(int))
        for start in range(0, len(order), training["batch_size"]):
            indices = order[start : start + training["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            prediction = predict(model, tensors, graph, indices)
            loss = torch.mean(((prediction - tensors["target"][indices]) / tensors["scale"]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"stage": "final", "epoch": epoch, "train_loss": float(np.mean(losses))})
        if epoch == 1 or epoch % 10 == 0 or epoch == selected_epoch:
            print(f"{variant} seed={seed} final epoch={epoch}/{selected_epoch}", flush=True)
    _, prediction = evaluate(model, tensors, graph, np.arange(25))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": seed,
            "epochs": selected_epoch,
            "parameters": parameter_count,
            "variant": variant,
            "counts": VARIANTS[variant][0],
            "connectivity": VARIANTS[variant][1],
        },
        output / "checkpoint.pt",
    )
    np.savez_compressed(
        output / "predictions.npz", prediction=prediction, prior=tensors["prior"].cpu().numpy()
    )
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    (output / "COMPLETE.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "variant": variant,
                "epochs": selected_epoch,
                "validation_loss": best,
                "trainable_parameters": parameter_count,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{variant} seed={seed} complete epochs={selected_epoch}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    train(args.variant, args.seed, args.device)
