# Production-Ready Churn Prediction with Drift Monitoring

A production-oriented customer churn prediction project that combines:
- XGBoost-based churn scoring
- Robust data drift detection (numeric + categorical)
- Monitoring orchestration with alerting and report history
- Interactive Streamlit dashboard for drift and prediction insights

## What Has Been Added

### 1) Enhanced Drift Detection
Implemented in `src/drift_detector.py`:
- Numeric drift tests:
  - Z-score mean shift
  - Kolmogorov-Smirnov (KS) test
  - Population Stability Index (PSI)
- Categorical drift tests:
  - Proportion shift
  - Chi-square test
  - Population Stability Index (PSI)
- Severity levels:
  - LOW, MEDIUM, HIGH
- Overall statuses:
  - OK, WARNING, CRITICAL
- Practical production improvements:
  - Effect-size guardrail for chi-square escalation
  - Binary-feature handling for KS influence
  - JSON report export and readable console reports

### 2) Prediction Pipeline Integrated with Drift Detection
Implemented in `src/predict_with_drift.py`:
- Keeps core prediction behavior from `src/predict.py`
- Adds optional, built-in drift checks before/with predictions
- Adds drift metadata to outputs:
  - `drift_status`
  - `drift_warning`
- Supports:
  - Single prediction
  - Batch prediction
  - Drift-only checks
- Includes graceful fallback if drift detector is unavailable

### 3) Monitoring Orchestration
Implemented in `src/run_monitoring.py`:
- End-to-end monitoring run for new data batches
- Uses custom drift detector as the monitoring source
- Evaluates configurable alert policy:
  - alert on warning
  - alert on critical
- Persists combined monitoring reports to history
- CLI options for:
  - custom reference data
  - custom stats path
  - webhook config
  - retry and timeout tuning
  - alert cooldown windows

### 4) Alerting System
Implemented in `src/alerting.py`:
- Console alert emission
- Optional webhook delivery
- Retry support with backoff
- Cooldown-based duplicate suppression
- Alert fingerprinting and unique alert IDs
- Alert event log output:
  - `data/alerts/alert_events.jsonl`
- Optional state tracking for cooldown:
  - `data/alerts/alert_state.json`

### 5) Interactive Dashboard
Implemented in `src/dashboard.py`:
- Streamlit dashboard for drift status and trends
- Reads latest combined and standalone drift reports
- Highlights top drifted numeric/categorical features
- Displays alert status from monitoring artifacts
- Includes richer visual design and summary cards

### 6) Documentation and Notebook Support
- `DRIFT_DETECTION_UPDATE.md`: production drift detector updates and outcomes
- `PREDICTION_DRIFT_INTEGRATION.md`: prediction + drift integration details
- Notebooks:
  - `notebooks/02_drift_detection.ipynb`
  - `notebooks/03_single_and_batch_prediction.ipynb`
  - `notebooks/04_drift_visualization.ipynb`

## Project Structure

Key folders and files:
- `src/`
  - `predict.py`
  - `predict_with_drift.py`
  - `drift_detector.py`
  - `run_monitoring.py`
  - `alerting.py`
  - `dashboard.py`
- `models/`
  - `xgb_churn_model.joblib`
  - `feature_names.joblib`
  - `model_metadata.json`
- `data/`
  - `processed/training_reference_stats.json`
  - `drift_history/` (generated drift and combined monitoring reports)
  - `alerts/` (generated alert logs/state)
  - `predictions/` (prediction artifacts)
  - `user_testing/` (sample baseline/drifted input files)
- `notebooks/` (analysis, integration checks, visualization)

## Installation

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### A) Run Predictions (Core)

```bash
python src/predict.py
```

### B) Run Predictions with Drift Detection

```bash
python src/predict_with_drift.py
```

### C) Run Monitoring on a Batch

```bash
python src/run_monitoring.py --data data/user_testing/baseline_large.csv
python src/run_monitoring.py --data data/user_testing/drifted_data.csv
```

Alerting examples:

```bash
python src/run_monitoring.py --data data/user_testing/drifted_data.csv --alert-on critical
python src/run_monitoring.py --data data/user_testing/drifted_data.csv --webhook-url https://example/webhook
python src/run_monitoring.py --data data/user_testing/drifted_data.csv --alert-cooldown-minutes 30
```

### D) Launch Dashboard

```bash
streamlit run src/dashboard.py
```

## Output Artifacts

Generated during monitoring/prediction workflows:
- Drift reports:
  - `data/drift_history/drift_report_<timestamp>.json`
- Combined monitoring reports:
  - `data/drift_history/combined_monitoring_report_<batch_id>.json`
- Alert log:
  - `data/alerts/alert_events.jsonl`
- Alert cooldown state (if enabled):
  - `data/alerts/alert_state.json`

## Dependencies

Defined in `requirements.txt`, including:
- Core ML/Data: pandas, numpy, scikit-learn, scipy, xgboost, joblib
- Monitoring/UI: streamlit, plotly
- API/Utilities: fastapi, uvicorn, pydantic, requests, python-dotenv
- Dev: jupyter, pytest, black, flake8

Note: `requirements.txt` still includes Evidently, while monitoring currently runs on the custom drift detector pipeline.

## Current Status

- Drift detection: integrated and enhanced for numeric/categorical shifts
- Prediction pipeline: drift-aware module available
- Monitoring: batch execution + combined report generation
- Alerting: console + webhook + cooldown suppression
- Dashboard: operational for drift history and alert visualization

## Related Docs

- `DRIFT_DETECTION_UPDATE.md`
- `PREDICTION_DRIFT_INTEGRATION.md`
- `src/README.md`
