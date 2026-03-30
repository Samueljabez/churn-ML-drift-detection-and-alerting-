# Production Prediction Module

## Overview

The `predict.py` module provides a production-ready interface for making churn predictions using the trained XGBoost model. It handles data preprocessing, feature encoding, validation, and delivers actionable risk assessments.

## Features

### ✓ Automatic Data Preprocessing
- Handles raw categorical data (auto one-hot encodes)
- Converts numeric strings to proper types
- Fills missing features with sensible defaults (0)
- Aligns features with training data structure

### ✓ Flexible Input Formats
- **Single predictions**: Dict or pandas Series
- **Batch predictions**: DataFrame with multiple customers
- **Raw data**: Categorical values (auto-encoded)
- **Pre-encoded data**: Binary features (direct prediction)

### ✓ Comprehensive Output
- `predicted_churn`: Binary prediction (0/1)
- `churn_probability`: Confidence score (0-1)
- `risk_level`: Business-friendly classification (LOW/MEDIUM/HIGH)
- `recommendation`: Actionable guidance for each customer
- `model_confidence`: Certainty measure of prediction

### ✓ Error Handling
- Validates input data structure
- Auto-fills missing encoded features
- Handles missing values gracefully
- Provides clear error messages

---

## Quick Start

### Installation
```python
# Import the predictor
from predict import ChurnPredictor
import pandas as pd

# Initialize (auto-locates model in ../models/)
predictor = ChurnPredictor()
```

### Single Customer Prediction
```python
# Raw categorical data (will be auto-encoded)
customer = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'tenure': 2,
    'PhoneService': 'Yes',
    'InternetService': 'Fiber optic',
    'MonthlyCharges': 95.50,
    'TotalCharges': 95.50,
    # ... other features
}

result = predictor.predict_single(customer)
print(result)
# Output:
# {
#   'churn_prediction': 1,
#   'churn_label': 'Yes',
#   'churn_probability': 0.9190,
#   'risk_level': 'HIGH',
#   'recommendation': 'URGENT - Offer retention incentive (discount/upgrade)',
#   'model_confidence': 0.5810
# }
```

### Batch Prediction
```python
# Load data
df = pd.read_csv('../data/user_testing/baseline.csv')

# Make predictions (raw_data=True for categorical, False for pre-encoded)
results = predictor.predict_batch(df, raw_data=True)

# View results
print(results[['customerID', 'predicted_churn', 'churn_probability', 'risk_level']])

# High-risk customers only
high_risk = results[results['risk_level'] == 'HIGH']
print(f"Found {len(high_risk)} high-risk customers")
```

### Convenience Function
```python
from predict import load_and_predict

# Load CSV, predict, and save results
results_df = load_and_predict(
    input_file='customer_data.csv',
    output_file='predictions.csv',
    raw_data=True
)
```

---

## API Reference

### `ChurnPredictor(model_dir=None)`

**Parameters:**
- `model_dir` (str, optional): Path to model directory. If None, auto-locates relative to script.

**Methods:**

#### `predict_single(data) -> Dict`
Make a prediction for a single customer.

**Parameters:**
- `data` (Dict or pd.Series): Customer features

**Returns:** Dictionary with prediction results

**Raises:** ValueError if features are invalid

---

#### `predict_batch(data, raw_data=False) -> DataFrame`
Make predictions for multiple customers.

**Parameters:**
- `data` (pd.DataFrame): Customer records
- `raw_data` (bool): If True, auto-encodes categorical features

**Returns:** DataFrame with original data + predictions

**Raises:** ValueError if data is empty

---

#### `get_model_info() -> Dict`
Return model metadata and configuration.

**Returns:** Dictionary with model details

---

### Utility Functions

#### `load_and_predict(input_file, output_file=None, raw_data=True) -> DataFrame`
Convenience function to load CSV, predict, and save.

**Parameters:**
- `input_file` (str): Path to input CSV
- `output_file` (str, optional): Path to save results
- `raw_data` (bool): Whether data has categorical features

**Returns:** Results DataFrame

---

## Data Format Examples

### Raw Categorical Format
```csv
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,InternetService,MonthlyCharges,TotalCharges,Churn
TEST-001,Female,0,Yes,No,12,Yes,DSL,50.50,606.00,No
TEST-002,Male,1,No,No,45,Yes,Fiber optic,85.25,3836.25,No
```

### Pre-Encoded Format (30 features after one-hot encoding)
```python
{
    'SeniorCitizen': 0,
    'tenure': 2,
    'MonthlyCharges': 95.50,
    'TotalCharges': 95.50,
    'PhoneService_Yes': 1,
    'MultipleLines_Yes': 1,
    'InternetService_Fiber optic': 1,
    'InternetService_No': 0,
    # ... 22 more binary features
}
```

---

## Output Interpretation

### Risk Levels
- **HIGH**: Churn probability > 70% → URGENT action needed
- **MEDIUM**: 50% < probability ≤ 70% → Monitor closely
- **LOW**: Probability ≤ 50% → Standard service

### Model Confidence
- Measures certainty (1 - |probability - 0.5|)
- 0.5 = barely confident (probability near 50%)
- 1.0 = highly confident (probability near 0% or 100%)

---

## Error Handling

The module handles common issues gracefully:

```python
# Missing input data
try:
    results = predictor.predict_batch(empty_df)
except ValueError as e:
    print(f"Error: {e}")

# Invalid model path
try:
    predictor = ChurnPredictor(model_dir='/wrong/path')
except FileNotFoundError as e:
    print(f"Model not found: {e}")
```

---

## Testing

Run the module directly to test with sample data:

```bash
python src/predict.py
```

This will:
1. Load the baseline test data
2. Make batch predictions on 10 test customers
3. Make a single prediction on an example customer
4. Display sample results and summary statistics

---

## Performance Notes

- **Single prediction**: ~100ms per customer
- **Batch prediction**: ~50ms for 100 customers
- **Memory**: ~500MB (model + features)

---

## Next Steps

1. **Integration**: Use in production API (Flask, FastAPI, etc.)
2. **Monitoring**: Track prediction accuracy over time
3. **Drift Detection**: Compare incoming data to training baseline
4. **Alerts**: Trigger when high-risk customers are identified
5. **A/B Testing**: Measure impact of retention interventions

---

For questions or issues, refer to the main project README or contact the ML team.

---

## Drift Monitoring with Evidently (Manual CLI)

Use the monitoring runner to execute custom drift detection and alerting in one command:

```bash
python src/run_monitoring.py --data data/user_testing/baseline_large.csv
python src/run_monitoring.py --data data/user_testing/drifted_data.csv
```

Optional flags:

```bash
# Alert only on CRITICAL drift
python src/run_monitoring.py --data new_batch.csv --alert-on critical

# Send alerts to webhook
python src/run_monitoring.py --data new_batch.csv --webhook-url https://example/webhook

# Disable alert dispatch
python src/run_monitoring.py --data new_batch.csv --disable-alerts

# Override reference data and stats
python src/run_monitoring.py --data new_batch.csv --reference data/user_testing/baseline_large.csv --stats data/processed/training_reference_stats.json
```

Generated artifacts:
- `data/drift_history/drift_report_<timestamp>.json`
- `data/drift_history/combined_monitoring_report_<batch_id>.json`

Dashboard:

```bash
streamlit run src/dashboard.py
```

The dashboard now reads combined monitoring history and surfaces alert levels, triggered events, and drift trends.
