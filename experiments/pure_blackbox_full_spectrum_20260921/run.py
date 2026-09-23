"""Train one requested pure-black-box model/seed on the locked split."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from model import BASGTDataK3, FiveBandBaseline
from settings import BANDS, CODE_DIR, INPUT_DIR, MODELS, RESULT_DIR, SEEDS, SEGMENT_COUNTS, TRAINING
from sklearn.linear_model import Ridge

from paper2_final.reproducibility import set_seed

FROZEN_GRAPH = CODE_DIR.parent / "v3_graph_all_band_auto_count_20260919" / "model.py"
graph_spec = importlib.util.spec_from_file_location("frozen_graph_topology", FROZEN_GRAPH)
frozen_graph = importlib.util.module_from_spec(graph_spec)
graph_spec.loader.exec_module(frozen_graph)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_boundaries(stage: str) -> list[np.ndarray]:
    return [
        np.load(RESULT_DIR / "segments" / f"{stage}_b{band}_k{count}_segments.npz")["bounds_m"]
        for band, count in enumerate(SEGMENT_COUNTS)
    ]


def stage_data(stage: str, model_name: str, device: str):
    prepared = np.load(INPUT_DIR / f"{stage}_NO_WB_ALL.npz")
    target_values = np.load(INPUT_DIR / "target.npz")
    fit = prepared["fit"].astype(int)
    target = target_values["target"].astype(np.float32)
    if model_name == "BASGT-DATA-K3":
        center = np.zeros(5, dtype=np.float32)
        scale = np.maximum(np.sqrt(np.mean(target[fit] ** 2, axis=(0, 1))), 1e-3).astype(np.float32)
        boundaries = load_boundaries(stage)
        graph = frozen_graph.graph_topology(boundaries, device)
    else:
        center = target[fit].mean(axis=(0, 1)).astype(np.float32)
        scale = np.maximum(target[fit].std(axis=(0, 1)), 1e-3).astype(np.float32)
        boundaries = None
        graph = None
    tensors = {
        "features": torch.tensor(
            prepared["features"][..., :83].copy(), device=device, dtype=torch.float32
        ),
        "target": torch.tensor(target, device=device, dtype=torch.float32),
        "center": torch.tensor(center, device=device),
        "scale": torch.tensor(scale, device=device),
    }
    metadata = {
        "fit": fit,
        "val": prepared["val"].astype(int),
        "dates": target_values["dates"].astype(str),
        "mileage_m": target_values["mileage_m"],
        "idx_dev": target_values["idx_dev"].astype(int),
        "center": center,
        "scale": scale,
        "boundaries": boundaries,
    }
    return metadata, tensors, graph


def build_model(model_name: str, boundaries, device: str):
    if model_name == "BASGT-DATA-K3":
        return BASGTDataK3(83, boundaries).to(device)
    family = model_name.removesuffix("-DATA")
    return FiveBandBaseline(family, 83).to(device)


def raw_prediction(model, model_name, features, graph):
    if model_name == "BASGT-DATA-K3":
        return model(features, *graph)
    return model(features)


def evaluate(model, model_name, tensors, graph, indices):
    model.eval()
    raw_outputs = []
    with torch.no_grad():
        for start in range(0, len(indices), TRAINING["batch_size"]):
            ids = indices[start : start + TRAINING["batch_size"]]
            raw_outputs.append(raw_prediction(model, model_name, tensors["features"][ids], graph))
    raw = torch.cat(raw_outputs)
    normalized_target = (tensors["target"][indices] - tensors["center"]) / tensors["scale"]
    loss = torch.mean((raw - normalized_target) ** 2)
    prediction = tensors["center"] + tensors["scale"] * raw
    return float(loss), prediction.cpu().numpy()


def fit_epochs(
    model,
    model_name,
    tensors,
    graph,
    fit,
    validation,
    seed,
    learning_rate,
    maximum_epochs,
    select,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=TRAINING["weight_decay"],
    )
    generator = np.random.default_rng(seed)
    best, selected_epoch, stale = float("inf"), 0, 0
    history = []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        losses = []
        order = generator.permutation(fit)
        for start in range(0, len(order), TRAINING["batch_size"]):
            ids = order[start : start + TRAINING["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            raw = raw_prediction(model, model_name, tensors["features"][ids], graph)
            normalized_target = (tensors["target"][ids] - tensors["center"]) / tensors["scale"]
            loss = torch.mean((raw - normalized_target) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), TRAINING["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_loss = None
        if select:
            validation_loss, _ = evaluate(model, model_name, tensors, graph, validation)
            if validation_loss < best - TRAINING["min_delta"]:
                best, selected_epoch, stale = validation_loss, epoch, 0
            else:
                stale += 1
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
            }
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == maximum_epochs:
            print(
                f"{model_name} seed={seed} epoch={epoch} train={np.mean(losses):.8f} "
                f"validation={validation_loss}",
                flush=True,
            )
        if select and stale >= TRAINING["patience"]:
            break
    return history, {
        "learning_rate": learning_rate,
        "epochs": selected_epoch,
        "validation_loss": best,
    }


def save_predictions(model_name: str, seed: int, prediction: np.ndarray, metadata) -> None:
    target_values = np.load(INPUT_DIR / "target.npz")
    target = target_values["target"]
    folder = RESULT_DIR / "predictions" / model_name / str(seed)
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        folder / "predictions.npz",
        predicted_bands=prediction,
        true_bands=target,
        predicted_full_profile=prediction.sum(axis=2),
        true_full_profile=target_values["absolute"],
        dates=target_values["dates"],
        mileage_m=target_values["mileage_m"],
        model_name=np.asarray(model_name),
        random_seed=np.asarray(seed),
        development_indices=target_values["idx_dev"],
    )


def train_neural(model_name: str, seed: int, device: str) -> None:
    selection, tensors, graph = stage_data("selection", model_name, device)
    learning_rates = (
        [TRAINING["basgt_learning_rate"]]
        if model_name == "BASGT-DATA-K3"
        else TRAINING["learning_rates"]
    )
    candidates = []
    history = []
    for learning_rate in learning_rates:
        set_seed(seed)
        model = build_model(model_name, selection["boundaries"], device)
        rows, selected = fit_epochs(
            model,
            model_name,
            tensors,
            graph,
            selection["fit"],
            selection["val"],
            seed,
            learning_rate,
            TRAINING["max_epochs"],
            True,
        )
        history.extend(
            {"stage": "selection", "candidate_learning_rate": learning_rate, **row} for row in rows
        )
        candidates.append(selected)
        del model
        torch.cuda.empty_cache()
    choice = min(candidates, key=lambda item: item["validation_loss"])
    del tensors, graph
    torch.cuda.empty_cache()

    final, tensors, graph = stage_data("final", model_name, device)
    set_seed(seed)
    model = build_model(model_name, final["boundaries"], device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    rows, _ = fit_epochs(
        model,
        model_name,
        tensors,
        graph,
        final["fit"],
        final["val"],
        seed,
        choice["learning_rate"],
        choice["epochs"],
        False,
    )
    history.extend({"stage": "final", "candidate_learning_rate": None, **row} for row in rows)
    _, prediction = evaluate(model, model_name, tensors, graph, np.arange(25))

    checkpoint_folder = RESULT_DIR / "checkpoints" / model_name / str(seed)
    history_folder = RESULT_DIR / "histories" / model_name / str(seed)
    checkpoint_folder.mkdir(parents=True, exist_ok=False)
    history_folder.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": model_name,
            "seed": seed,
            "selected": choice,
            "selection_candidates": candidates,
            "target_center": final["center"],
            "target_scale": final["scale"],
            "parameter_count": parameter_count,
            "feature_channels": list(range(83)),
            "whitebox_prior_used": False,
            "post_prediction_smoothing": False,
        },
        checkpoint_folder / "checkpoint.pt",
    )
    pd.DataFrame(history).to_csv(history_folder / "training_history.csv", index=False)
    save_predictions(model_name, seed, prediction, final)
    write_json(
        checkpoint_folder / "COMPLETE.json",
        {
            "model": model_name,
            "seed": seed,
            "selected": choice,
            "parameter_count": parameter_count,
        },
    )
    print(f"COMPLETE {model_name} seed={seed}", flush=True)


def train_ridge() -> None:
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    selection, tensors, _ = stage_data("selection", "Ridge-DATA", "cpu")
    selection_features = tensors["features"].numpy()
    selection_target = ((tensors["target"] - tensors["center"]) / tensors["scale"]).numpy()
    fit, validation = selection["fit"], selection["val"]
    choices, history = [], []
    for band, band_name in enumerate(BANDS):
        features = selection_features[:, :, band].reshape(25, -1).astype(np.float64)
        target = selection_target[:, :, band].astype(np.float64)
        kernel_scale = max(
            float(np.mean(np.sum((features[fit] - features[fit].mean(0)) ** 2, axis=1))),
            1.0,
        )
        normalized = features / np.sqrt(kernel_scale)
        candidates = []
        for alpha in alphas:
            estimator = Ridge(alpha=alpha, solver="svd").fit(normalized[fit], target[fit])
            prediction = estimator.predict(normalized[validation])
            loss = float(np.mean((prediction - target[validation]) ** 2))
            candidates.append({"alpha": alpha, "validation_loss": loss})
            history.append(
                {
                    "stage": "selection",
                    "band": band_name,
                    "alpha": alpha,
                    "validation_loss": loss,
                }
            )
        choices.append(min(candidates, key=lambda item: item["validation_loss"]))

    final, tensors, _ = stage_data("final", "Ridge-DATA", "cpu")
    final_features = tensors["features"].numpy()
    final_target = ((tensors["target"] - tensors["center"]) / tensors["scale"]).numpy()
    fit = final["fit"]
    normalized_prediction = np.empty((25, 1433, 5), dtype=np.float64)
    states = []
    parameter_count = 0
    for band, choice in enumerate(choices):
        features = final_features[:, :, band].reshape(25, -1).astype(np.float64)
        target = final_target[:, :, band].astype(np.float64)
        input_mean = features[fit].mean(0)
        target_mean = target[fit].mean(0)
        kernel_scale = max(float(np.mean(np.sum((features[fit] - input_mean) ** 2, axis=1))), 1.0)
        normalized = (features - input_mean) / np.sqrt(kernel_scale)
        train_features = normalized[fit]
        dual = np.linalg.solve(
            train_features @ train_features.T + choice["alpha"] * np.eye(len(fit)),
            target[fit] - target_mean,
        )
        normalized_prediction[:, :, band] = normalized @ train_features.T @ dual + target_mean
        parameter_count += dual.size + target_mean.size
        states.append(
            {
                "input_mean": input_mean,
                "kernel_scale": kernel_scale,
                "train_features": train_features,
                "dual": dual,
                "target_mean": target_mean,
                "alpha": choice["alpha"],
            }
        )
    prediction = final["center"] + final["scale"] * normalized_prediction
    checkpoint_folder = RESULT_DIR / "checkpoints" / "Ridge-DATA" / "0"
    history_folder = RESULT_DIR / "histories" / "Ridge-DATA" / "0"
    checkpoint_folder.mkdir(parents=True, exist_ok=False)
    history_folder.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "model_name": "Ridge-DATA",
            "seed": 0,
            "bands": states,
            "selected": choices,
            "target_center": final["center"],
            "target_scale": final["scale"],
            "parameter_count_definition": "dual coefficients plus fitted target intercepts",
            "parameter_count": parameter_count,
            "feature_channels": list(range(83)),
            "whitebox_prior_used": False,
        },
        checkpoint_folder / "checkpoint.pt",
    )
    pd.DataFrame(history).to_csv(history_folder / "training_history.csv", index=False)
    save_predictions("Ridge-DATA", 0, prediction, final)
    write_json(
        checkpoint_folder / "COMPLETE.json",
        {
            "model": "Ridge-DATA",
            "seed": 0,
            "selected": choices,
            "parameter_count": parameter_count,
        },
    )
    print("COMPLETE Ridge-DATA seed=0", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    if arguments.model == "Ridge-DATA":
        if arguments.seed != 0:
            raise ValueError("Ridge-DATA uses deterministic seed 0")
        train_ridge()
        return
    if arguments.seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}")
    train_neural(arguments.model, arguments.seed, arguments.device)


if __name__ == "__main__":
    main()
