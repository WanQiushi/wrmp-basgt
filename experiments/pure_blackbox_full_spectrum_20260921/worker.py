"""Run a fixed half of the model/seed matrix on one GPU."""

from __future__ import annotations

import argparse
import subprocess
import sys

from settings import CODE_DIR

TASKS = {
    0: [
        ("Ridge-DATA", 0),
        ("BASGT-DATA-K3", 20260901),
        ("BASGT-DATA-K3", 20260903),
        ("BASGT-DATA-K3", 20260905),
        ("TimeXer-DATA", 20260902),
        ("TimeXer-DATA", 20260904),
        ("iTransformer-DATA", 20260901),
        ("iTransformer-DATA", 20260903),
        ("LSTM-DATA", 20260902),
        ("LSTM-DATA", 20260904),
        ("DLinear-DATA", 20260901),
        ("DLinear-DATA", 20260903),
        ("DLinear-DATA", 20260905),
    ],
    1: [
        ("BASGT-DATA-K3", 20260902),
        ("BASGT-DATA-K3", 20260904),
        ("TimeXer-DATA", 20260901),
        ("TimeXer-DATA", 20260903),
        ("TimeXer-DATA", 20260905),
        ("iTransformer-DATA", 20260902),
        ("iTransformer-DATA", 20260904),
        ("iTransformer-DATA", 20260905),
        ("LSTM-DATA", 20260901),
        ("LSTM-DATA", 20260903),
        ("LSTM-DATA", 20260905),
        ("DLinear-DATA", 20260902),
        ("DLinear-DATA", 20260904),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device", required=True)
    arguments = parser.parse_args()
    print(
        f"WORKER_START worker={arguments.worker} device={arguments.device} "
        f"tasks={len(TASKS[arguments.worker])}",
        flush=True,
    )
    for model, seed in TASKS[arguments.worker]:
        print(f"TASK_START model={model} seed={seed}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(CODE_DIR / "run.py"),
                "--model",
                model,
                "--seed",
                str(seed),
                "--device",
                arguments.device,
            ],
            check=True,
        )
        print(f"TASK_COMPLETE model={model} seed={seed}", flush=True)
    print(f"WORKER_COMPLETE worker={arguments.worker}", flush=True)


if __name__ == "__main__":
    main()
