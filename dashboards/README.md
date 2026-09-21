# Dashboards

This folder contains the Streamlit interfaces used to annotate recorded poses
and inspect evaluation results for all three motion-imitation methods.

## Setup

Install the dashboard dependencies in the `media` environment:

```powershell
conda activate media
python -m pip install streamlit pandas altair
```

Run all commands from the repository root.

## Dataset annotation dashboard

```powershell
streamlit run dashboards/annotation-dashboard/dataset_annotation_app.py
```

This dashboard loads videos from `dataset/`, displays frames and pose
annotations, and saves manual corrections to `annotations_filled.json`.

## Evaluation results dashboard

```powershell
streamlit run dashboards/streamlit-results-dashboard/evaluation_results_app.py
```

This dashboard reads:

```text
dashboards/results/ik_method_metrics.json
dashboards/results/performance_metrics.json
```

Generate these result files with:

```powershell
python -m server.common.evaluation.evaluate_ik_methods dataset
python -m server.common.evaluation.evaluate_performance_metrics dataset
```

The `results/` folder also contains the demo videos linked from the main
README.

## Related documentation

- [Shared evaluation tools](../server/common/evaluation/README.md)
- [Root setup and execution](../README.md)
