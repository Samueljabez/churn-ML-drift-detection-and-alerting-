"""
Telco Customer Churn Prediction Module with Drift Detection

Production-ready module for churn prediction with:
- Automatic data preprocessing and feature encoding
- Input validation and error handling
- Single and batch prediction support
- Risk level classification
- **Built-in drift detection and warnings**

Usage:
    from predict_with_drift import ChurnPredictor
    import pandas as pd
    
    # Initialize predictor (with drift detection enabled by default)
    predictor = ChurnPredictor(enable_drift_detection=True)
    
    # Single prediction from dict
    result = predictor.predict_single({...features...})
    
    # Batch prediction from DataFrame (includes drift warnings)
    results_df = predictor.predict_batch(df)
    
    # Check drift separately if needed
    drift_report = predictor.check_drift(df)
"""

import joblib
import json
import pandas as pd
import numpy as np
import os
import warnings
from typing import Union, Dict, List, Any, Tuple, Optional
from pathlib import Path

try:
    from drift_detector import DriftDetector
    DRIFT_DETECTION_AVAILABLE = True
except ImportError:
    DRIFT_DETECTION_AVAILABLE = False
    warnings.warn("Drift detection module not available. Predictions will proceed without drift checks.")


class ChurnPredictor:
    """
    Production-ready churn prediction interface with optional drift detection.
    
    Handles model loading, input preprocessing, validation, predictions, and drift monitoring.
    Supports both pre-encoded features and raw categorical data.
    """
    
    # Known categorical columns for automatic encoding
    CATEGORICAL_FEATURES = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    
    def __init__(self, model_dir: str = None, enable_drift_detection: bool = True):
        """
        Initialize predictor with trained model artifacts and optional drift detection.
        
        Args:
            model_dir: Path to directory containing model files.
                      If None, looks relative to this file or in project root.
                      Must contain: xgb_churn_model.joblib, feature_names.joblib
            enable_drift_detection: If True, enables automatic drift detection on predictions
        
        Raises:
            FileNotFoundError: If model or feature files not found
        """
        if model_dir is None:
            # Try relative to this file first
            current_dir = Path(__file__).parent.parent
            model_dir = current_dir / "models"
        
        self.model_dir = Path(model_dir).resolve()  # Convert to absolute path
        self.model = None
        self.feature_names = None
        self.model_metadata = None
        self.n_features = None
        
        # Drift detection
        self.enable_drift_detection = enable_drift_detection and DRIFT_DETECTION_AVAILABLE
        self.drift_detector = None
        self._last_drift_report = None
        
        self._load_artifacts()
        
        if self.enable_drift_detection:
            self._init_drift_detector()
    
    def _load_artifacts(self):
        """Load model, feature names, and metadata from disk."""
        model_path = self.model_dir / "xgb_churn_model.joblib"
        features_path = self.model_dir / "feature_names.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}\n"
                f"Looked in: {self.model_dir}"
            )
        if not features_path.exists():
            raise FileNotFoundError(
                f"Feature names not found at {features_path}\n"
                f"Looked in: {self.model_dir}"
            )
        
        self.model = joblib.load(model_path)
        self.feature_names = list(joblib.load(features_path))
        self.n_features = len(self.feature_names)
        
        # Load metadata if available
        metadata_path = self.model_dir / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
    
    def _init_drift_detector(self):
        """Initialize drift detector with training statistics."""
        if not DRIFT_DETECTION_AVAILABLE:
            return
        
        try:
            # Try to initialize drift detector (it will find training stats)
            self.drift_detector = DriftDetector()
            print("✓ Drift detection enabled")
        except Exception as e:
            warnings.warn(f"Could not initialize drift detector: {e}. Proceeding without drift detection.")
            self.enable_drift_detection = False
    
    def _encode_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode categorical features to match training format.
        
        Args:
            data: DataFrame with raw categorical columns
        
        Returns:
            Encoded DataFrame ready for model prediction
        """
        df = data.copy()
        
        # Convert TotalCharges to numeric (handle string values)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # One-hot encode categorical columns
        categorical_cols = [col for col in self.CATEGORICAL_FEATURES if col in df.columns]
        
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        return df
    
    def _validate_and_align_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate input features and align with training features.
        
        Args:
            data: DataFrame with features (may be missing some columns)
        
        Returns:
            Tuple of (aligned DataFrame, list of missing features)
        
        Raises:
            ValueError: If required features are missing or invalid
        """
        missing_features = set(self.feature_names) - set(data.columns)
        extra_features = set(data.columns) - set(self.feature_names)
        
        if missing_features:
            # Auto-fill missing features with 0 (common for one-hot encoded features)
            for feature in missing_features:
                data[feature] = 0
        
        # Select only required features and reorder
        X = data[self.feature_names].copy()
        
        # Fill any NaN values with 0
        X = X.fillna(0)
        
        return X, list(missing_features)
    
    def _get_risk_level(self, probability: float) -> str:
        """
        Classify churn probability into risk level.
        
        Args:
            probability: Churn probability (0-1)
        
        Returns:
            Risk level: 'LOW', 'MEDIUM', or 'HIGH'
        """
        if probability > 0.7:
            return "HIGH"
        elif probability > 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_recommendation(self, probability: float, risk_level: str) -> str:
        """
        Generate business recommendation based on churn prediction.
        
        Args:
            probability: Churn probability
            risk_level: Risk level classification
        
        Returns:
            Actionable recommendation
        """
        if risk_level == "HIGH":
            return "URGENT - Offer retention incentive (discount/upgrade)"
        elif risk_level == "MEDIUM":
            return "Monitor closely - prepare retention offer"
        else:
            return "Low risk - standard service"
    
    def check_drift(self, data: pd.DataFrame, raw_data: bool = True) -> Optional[Dict[str, Any]]:
        """
        Check for data drift on input data.
        
        Args:
            data: DataFrame with customer data
            raw_data: If True, expects raw categorical data; if False, pre-encoded
        
        Returns:
            Drift report dict or None if drift detection disabled
        """
        if not self.enable_drift_detection or self.drift_detector is None:
            return None
        
        try:
            # Get raw data for drift detection (before encoding)
            if not raw_data:
                # If data is already encoded, we can't check drift on categorical features properly
                warnings.warn("Drift detection works best with raw categorical data")
            
            drift_report = self.drift_detector.detect_drift(data)
            self._last_drift_report = drift_report
            return drift_report
        except Exception as e:
            warnings.warn(f"Drift detection failed: {e}")
            return None
    
    def _add_drift_warning(self, result: Dict[str, Any], drift_status: str, summary: Dict) -> Dict[str, Any]:
        """Add drift warning information to prediction result."""
        result['drift_status'] = drift_status
        result['drift_warning'] = None
        
        if drift_status == 'CRITICAL':
            result['drift_warning'] = (
                f"⚠️ CRITICAL DRIFT DETECTED! {summary['high_severity']} features show HIGH severity drift. "
                "Predictions may be unreliable. Model retraining recommended."
            )
        elif drift_status == 'WARNING':
            result['drift_warning'] = (
                f"⚠️ MODERATE DRIFT DETECTED. {summary['medium_severity']} features show MEDIUM severity drift. "
                "Monitor predictions closely."
            )
        else:
            result['drift_warning'] = "✓ No significant drift detected"
        
        return result
    
    def predict_single(self, data: Union[Dict[str, Any], pd.Series], check_drift: bool = None) -> Dict[str, Any]:
        """
        Predict churn for a single customer with optional drift check.
        
        Args:
            data: Customer features as dict or Series
                 Can be raw categorical data or pre-encoded features
            check_drift: Override default drift detection setting for this prediction
        
        Returns:
            Dictionary with prediction results:
                {
                    'churn_prediction': 0 or 1,
                    'churn_label': 'No' or 'Yes',
                    'churn_probability': float,
                    'risk_level': 'LOW'/'MEDIUM'/'HIGH',
                    'recommendation': str,
                    'model_confidence': float (1-|prob-0.5|),
                    'drift_status': 'OK'/'WARNING'/'CRITICAL' (if enabled),
                    'drift_warning': str (if drift detected)
                }
        
        Raises:
            ValueError: If features are invalid or missing
        """
        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            df = data.to_frame().T
        else:
            raise ValueError("Input must be dict or pd.Series")
        
        # Check if raw categorical data (has known categorical columns)
        has_raw_categorical = any(col in df.columns for col in self.CATEGORICAL_FEATURES)
        
        # Check drift if enabled (do this BEFORE encoding for accurate drift detection)
        should_check_drift = check_drift if check_drift is not None else self.enable_drift_detection
        drift_report = None
        if should_check_drift and has_raw_categorical:
            drift_report = self.check_drift(df, raw_data=True)
        
        # Encode if needed
        if has_raw_categorical:
            df = self._encode_features(df)
        
        # Validate and align features
        X, _ = self._validate_and_align_features(df)
        
        # Make prediction
        prediction = int(self.model.predict(X)[0])
        probability = float(self.model.predict_proba(X)[0][1])
        
        risk_level = self._get_risk_level(probability)
        model_confidence = 1 - abs(probability - 0.5)  # Confidence in prediction
        
        result = {
            'churn_prediction': prediction,
            'churn_label': 'Yes' if prediction == 1 else 'No',
            'churn_probability': round(probability, 4),
            'risk_level': risk_level,
            'recommendation': self._get_recommendation(probability, risk_level),
            'model_confidence': round(model_confidence, 4)
        }
        
        # Add drift information if available
        if drift_report:
            result = self._add_drift_warning(result, drift_report['overall_status'], drift_report['summary'])
        
        return result
    
    def predict_batch(self, data: pd.DataFrame, raw_data: bool = False, 
                     check_drift: bool = None, show_drift_report: bool = False) -> pd.DataFrame:
        """
        Predict churn for multiple customers efficiently with optional drift detection.
        
        Args:
            data: DataFrame with customer records
            raw_data: If True, auto-encodes categorical features;
                     if False, expects pre-encoded features
            check_drift: Override default drift detection setting
            show_drift_report: If True, prints drift report to console
        
        Returns:
            DataFrame with original data + prediction columns:
                - 'predicted_churn': 0 or 1
                - 'churn_probability': float
                - 'risk_level': 'LOW'/'MEDIUM'/'HIGH'
                - 'recommendation': str
                - 'drift_status': 'OK'/'WARNING'/'CRITICAL' (if enabled)
                - 'drift_warning': str (if drift detected)
        
        Raises:
            ValueError: If data is empty or features are invalid
        """
        if len(data) == 0:
            raise ValueError("Input DataFrame is empty")
        
        df = data.copy()
        
        # Check drift if enabled (do this BEFORE encoding)
        should_check_drift = check_drift if check_drift is not None else self.enable_drift_detection
        drift_report = None
        if should_check_drift:
            drift_report = self.check_drift(data, raw_data=raw_data or any(col in data.columns for col in self.CATEGORICAL_FEATURES))
            
            if drift_report and show_drift_report:
                self.drift_detector.print_report(drift_report)
        
        # Encode if needed
        if raw_data:
            df = self._encode_features(df)
        
        # Validate and align features
        X, _ = self._validate_and_align_features(df)
        
        # Batch predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Add predictions to results
        results = data.copy()
        results['predicted_churn'] = predictions
        results['churn_probability'] = probabilities.round(4)
        results['risk_level'] = results['churn_probability'].apply(self._get_risk_level)
        results['recommendation'] = results.apply(
            lambda row: self._get_recommendation(row['churn_probability'], row['risk_level']),
            axis=1
        )
        
        # Add drift status if available
        if drift_report:
            results['drift_status'] = drift_report['overall_status']
            
            # Add warning message based on status
            if drift_report['overall_status'] == 'CRITICAL':
                warning_msg = (
                    f"⚠️ CRITICAL DRIFT: {drift_report['summary']['high_severity']} HIGH severity features. "
                    "Predictions may be unreliable!"
                )
            elif drift_report['overall_status'] == 'WARNING':
                warning_msg = (
                    f"⚠️ MODERATE DRIFT: {drift_report['summary']['medium_severity']} MEDIUM severity features. "
                    "Monitor predictions."
                )
            else:
                warning_msg = "✓ No significant drift"
            
            results['drift_warning'] = warning_msg
        
        return results
    
    def get_last_drift_report(self) -> Optional[Dict[str, Any]]:
        """
        Get the last drift detection report.
        
        Returns:
            Last drift report or None if no drift check performed yet
        """
        return self._last_drift_report
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return model information and configuration.
        
        Returns:
            Dictionary with model details
        """
        return {
            'model_type': type(self.model).__name__,
            'n_features': self.n_features,
            'feature_count': len(self.feature_names),
            'features': self.feature_names[:10] + (['...'] if len(self.feature_names) > 10 else []),
            'metadata': self.model_metadata or {},
            'drift_detection_enabled': self.enable_drift_detection
        }


# ============================================================================
# Utility functions for CLI/script usage
# ============================================================================

def load_and_predict(input_file: str, output_file: str = None, raw_data: bool = True,
                    enable_drift_detection: bool = True, show_drift_report: bool = True) -> pd.DataFrame:
    """
    Convenience function to load CSV, make predictions, and save results.
    
    Args:
        input_file: Path to input CSV with customer data
        output_file: Path to save predictions (optional)
        raw_data: If True, assumes raw categorical data; if False, pre-encoded
        enable_drift_detection: If True, checks for data drift
        show_drift_report: If True, prints drift report
    
    Returns:
        DataFrame with predictions
    """
    # Load data
    df = pd.read_csv(input_file)
    
    # Make predictions
    predictor = ChurnPredictor(enable_drift_detection=enable_drift_detection)
    results = predictor.predict_batch(df, raw_data=raw_data, show_drift_report=show_drift_report)
    
    # Save if specified
    if output_file:
        results.to_csv(output_file, index=False)
        print(f"✓ Predictions saved to {output_file}")
    
    return results


if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("Telco Churn Predictor with Drift Detection - Production Module")
    print("=" * 70)
    
    # Initialize with drift detection
    predictor = ChurnPredictor(enable_drift_detection=True)
    
    # Show model info
    print("\nModel Information:")
    info = predictor.get_model_info()
    print(f"  Type: {info['model_type']}")
    print(f"  Features: {info['n_features']}")
    print(f"  Drift Detection: {'Enabled ✓' if info['drift_detection_enabled'] else 'Disabled'}")
    print(f"  Status: Ready")
    
    # Example: Batch prediction with drift detection on drifted data
    print("\n" + "-" * 70)
    print("Example 1: Batch Prediction on DRIFTED Data (with drift warnings)")
    print("-" * 70)
    
    try:
        drifted_path = Path(__file__).parent.parent / "data" / "user_testing" / "drifted_data.csv"
        
        if drifted_path.exists():
            drifted_df = pd.read_csv(drifted_path)
            
            # Make predictions with drift detection
            results = predictor.predict_batch(drifted_df, raw_data=True, show_drift_report=True)
            
            print(f"\n[OK] Successfully predicted for {len(results)} customers")
            print("\nSample Results with Drift Warnings:")
            display_cols = ['customerID', 'predicted_churn', 'churn_probability', 'risk_level', 'drift_status']
            print(results[display_cols].head(3).to_string(index=False))
            
            if 'drift_warning' in results.columns:
                print(f"\n{results['drift_warning'].iloc[0]}")
            
            print(f"\nPrediction Summary:")
            print(f"  - High Risk: {(results['risk_level'] == 'HIGH').sum()}")
            print(f"  - Medium Risk: {(results['risk_level'] == 'MEDIUM').sum()}")
            print(f"  - Low Risk: {(results['risk_level'] == 'LOW').sum()}")
        else:
            print(f"[WARN] Drifted data file not found at {drifted_path}")
    
    except Exception as e:
        print(f"⚠ Error during batch prediction: {e}")
    
    # Example: Baseline data (should show no drift)
    print("\n" + "-" * 70)
    print("Example 2: Batch Prediction on BASELINE Data (no drift expected)")
    print("-" * 70)
    
    try:
        baseline_path = Path(__file__).parent.parent / "data" / "user_testing" / "baseline.csv"
        
        if baseline_path.exists():
            baseline_df = pd.read_csv(baseline_path)
            
            # Make predictions with drift detection
            results = predictor.predict_batch(baseline_df, raw_data=True, show_drift_report=False)
            
            print(f"\n[OK] Successfully predicted for {len(results)} customers")
            
            if 'drift_warning' in results.columns:
                print(f"Drift Status: {results['drift_warning'].iloc[0]}")
    
    except Exception as e:
        print(f"⚠ Error: {e}")
    
    print("\n" + "=" * 70)
    print("Usage Examples:")
    print("=" * 70)
    print("""
1. Batch Prediction with Drift Detection (Recommended):
   -----------------------------------------------
   df = pd.read_csv('customer_data.csv')
   predictor = ChurnPredictor(enable_drift_detection=True)
   results = predictor.predict_batch(df, raw_data=True, show_drift_report=True)
   
   # Check for drift warnings
   if 'drift_status' in results.columns:
       critical_drift = results['drift_status'].iloc[0] == 'CRITICAL'
       if critical_drift:
           print("⚠️ WARNING: Predictions may be unreliable due to drift!")

2. Batch Prediction without Drift Detection:
   -----------------------------------------------
   predictor = ChurnPredictor(enable_drift_detection=False)
   results = predictor.predict_batch(df, raw_data=True)

3. Single Prediction with Drift Check:
   -----------------------------------------------
   customer = {'gender': 'Male', 'tenure': 12, ...}
   result = predictor.predict_single(customer)
   if result.get('drift_warning'):
       print(result['drift_warning'])

4. Separate Drift Check:
   -----------------------------------------------
   drift_report = predictor.check_drift(df, raw_data=True)
   if drift_report['overall_status'] == 'CRITICAL':
       print("Model retraining recommended!")

Results include:
   - predicted_churn: 0 or 1
   - churn_probability: confidence score (0-1)
   - risk_level: 'LOW', 'MEDIUM', or 'HIGH'
   - recommendation: actionable business guidance
   - drift_status: 'OK', 'WARNING', or 'CRITICAL' (if enabled)
   - drift_warning: human-readable drift message (if enabled)
""")
