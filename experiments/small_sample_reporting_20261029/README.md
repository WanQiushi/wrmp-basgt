# Small-sample reporting with a shared full-budget reference

The reporting view reuses the five frozen main-table WRMP--BASGT runs at 16
dates. It keeps the independently fitted 12/8/4-date results and all baseline
results unchanged. The mean/sample-standard-deviation endpoint is
**33.762450 / 0.056735 mm**. All five seeds are reused; this is not a choice of
the best seed or replacement of individual metric cells.

The separate small-sample n16 experiment remains **33.779141 / 0.056331 mm**.
It used the same target profiles, evaluation dates, inner split, and adaptive
boundaries, but rebuilt its prepared arrays and used a separate training
runner. The audit found small input differences and selected epoch counts of
41/36/27/25/40 rather than 24/31/38/30/38. It did not isolate a single cause
for the changed optimisation trajectory. The independent run is not relabelled
as the frozen main run.

Given the original private small-sample `metrics_by_seed.csv`, export the
separate reporting view:

```bash
python experiments/small_sample_reporting_20261029/export.py --small-sample-metrics /private/small_sample/metrics_by_seed.csv --output /private/reports/small_sample_main_endpoint
```

The command writes per-seed full-profile RMSE, seed means/sample standard
deviations, a source manifest, and the independent n16 reference. Its input
is never overwritten. The canonical endpoint is read from
`reference_results/k3_component_and_full_budget.json`, whose provenance is the
frozen K=3 main experiment. Do not describe the reused endpoint as a new
16-date run in the small-sample pipeline.

The original 16/12/8/4 experiment used one nested date-subset sequence with
12/4, 9/3, 6/2, and 3/1 internal fitting/validation splits. Five seeds quantify
training randomness, not repeated sampling of dates. The nine evaluation
dates have been reused during development. Private raw WRMP preparation and
the original small-sample training pipeline remain outside this source release.
