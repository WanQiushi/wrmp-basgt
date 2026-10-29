# Three-layer component ablation

These are the K=3 component controls used in the manuscript component-ablation
table. They are separate from the historical K=1 controls in
`v3_graph_four_band_ablation_20260919`.

The full model is the frozen `graph_k3` run from
`v3_graph_depth_ablation_20260919`. Each control retains three independently
parameterised graph layers and 4,230,604 registered trainable parameters.
Equal registered counts do not imply equal effective capacity after a route is
disabled. The 2026-10-29 audit inspected all 20 checkpoints (four models, five
seeds), their layer indices, completion records, and predictions.

| Variant | Operation |
| --- | --- |
| `k3_no_graph` | Replace graph weights with self loops; retain all three layers |
| `k3_no_seg` | Use a 1433-point adjacent graph instead of adaptive segment pooling |
| `k3_no_local_direct` | Zero the direct local route into fusion; retain the local encoder for graph tokens |

From the repository root, after preparing the same selection/final inputs and
adaptive boundaries used by the main K=3 model:

```bash
python experiments/v3_graph_k3_component_ablation_20260922/run.py --variant k3_no_graph --seed 20260901 --device cuda:0
python experiments/v3_graph_k3_component_ablation_20260922/run.py --variant k3_no_seg --seed 20260901 --device cuda:0
python experiments/v3_graph_k3_component_ablation_20260922/run.py --variant k3_no_local_direct --seed 20260901 --device cuda:0
```

Repeat for seeds 20260902--20260905. Run evaluation only after both the five
frozen `graph_k3` predictions and the 15 controls are available:

```bash
python experiments/v3_graph_k3_component_ablation_20260922/evaluate.py
```

The evaluator reads the shared fifth-band prior and reports per-seed metrics,
mean/sample standard deviation, paired pooled-RMSE changes, parameter counts,
and selected epochs. It does not use the K=1 controls.

Audited pooled sub-200 m RMSEs are 0.671503 (full), 0.696193 (no graph),
0.730890 (no segments), and 0.687012 mm (no direct local route). Checkpoint
metadata and the unrounded reference metrics are in
[`reference_results/k3_component_and_full_budget.json`](../../reference_results/k3_component_and_full_budget.json).
No restricted arrays or weights are distributed.
