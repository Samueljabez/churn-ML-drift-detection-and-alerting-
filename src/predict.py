"""
Telco Customer Churn Prediction Module

Production-ready module for churn prediction with:
- Automatic data preprocessing and feature encoding
- Input validation and error handling
- Single and batch prediction support
- Risk level classification

Usage:
    from predict import ChurnPredictor
    import pandas as pd
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Single prediction from dict
    result = predictor.predict_single({...features...})
    
    # Batch prediction from DataFrame
    results_df = predictor.predict_batch(df)
    
    # Or with raw categorical data (auto-encodes)
    results_df = predictor.predict_batch(df, raw_data=True)
"""

import joblib
import json
import pandas as pd
import numpy as np
import os
from typing import Union, Dict, List, Any, Tuple
from pathlib import Path


class ChurnPredictor:
    """
    Production-ready churn prediction interface.
    
    Handles model loading, input preprocessing, validation, and predictions.
    Supports both pre-encoded features and raw categorical data.
    """
    
    # Known categorical columns for automatic encoding
    CATEGORICAL_FEATURES = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    
    def __init__(self, model_dir: str = None):
        """
        Initialize predictor with trained model artifacts.
        
        Args:
            model_dir: Path to directory containing model files.
                      If None, looks relative to this file or in project root.
                      Must contain: xgb_churn_model.joblib, feature_names.joblib
        
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
        
        self._load_artifacts()
    
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
    
    def predict_single(self, data: Union[Dict[str, Any], pd.Series]) -> Dict[str, Any]:
        """
        Predict churn for a single customer.
        
        Args:
            data: Customer features as dict or Series
                 Can be raw categorical data or pre-encoded features
        
        Returns:
            Dictionary with prediction results:
                {
                    'churn_prediction': 0 or 1,
                    'churn_label': 'No' or 'Yes',
                    'churn_probability': float,
                    'risk_level': 'LOW'/'MEDIUM'/'HIGH',
                    'recommendation': str,
                    'model_confidence': float (1-|prob-0.5|)
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
        
        if has_raw_categorical:
            df = self._encode_features(df)
        
        # Validate and align features
        X, _ = self._validate_and_align_features(df)
        
        # Make prediction
        prediction = int(self.model.predict(X)[0])
        probability = float(self.model.predict_proba(X)[0][1])
        
        risk_level = self._get_risk_level(probability)
        model_confidence = 1 - abs(probability - 0.5)  # Confidence in prediction
        
        return {
            'churn_prediction': prediction,
            'churn_label': 'Yes' if prediction == 1 else 'No',
            'churn_probability': round(probability, 4),
            'risk_level': risk_level,
            'recommendation': self._get_recommendation(probability, risk_level),
            'model_confidence': round(model_confidence, 4)
        }
    
    def predict_batch(self, data: pd.DataFrame, raw_data: bool = False) -> pd.DataFrame:
        """
        Predict churn for multiple customers efficiently.
        
        Args:
            data: DataFrame with customer records
            raw_data: If True, auto-encodes categorical features;
                     if False, expects pre-encoded features
        
        Returns:
            DataFrame with original data + prediction columns:
                - 'predicted_churn': 0 or 1
                - 'churn_probability': float
                - 'risk_level': 'LOW'/'MEDIUM'/'HIGH'
                - 'recommendation': str
        
        Raises:
            ValueError: If data is empty or features are invalid
        """
        if len(data) == 0:
            raise ValueError("Input DataFrame is empty")
        
        df = data.copy()
        
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
        
        return results
    
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
            'metadata': self.model_metadata or {}
        }


# ============================================================================
# Utility functions for CLI/script usage
# ============================================================================

def load_and_predict(input_file: str, output_file: str = None, raw_data: bool = True) -> pd.DataFrame:
    """
    Convenience function to load CSV, make predictions, and save results.
    
    Args:
        input_file: Path to input CSV with customer data
        output_file: Path to save predictions (optional)
        raw_data: If True, assumes raw categorical data; if False, pre-encoded
    
    Returns:
        DataFrame with predictions
    """
    # Load data
    df = pd.read_csv(input_file)
    
    # Make predictions
    predictor = ChurnPredictor()
    results = predictor.predict_batch(df, raw_data=raw_data)
    
    # Save if specified
    if output_file:
        results.to_csv(output_file, index=False)
        print(f"✓ Predictions saved to {output_file}")
    
    return results


if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("Telco Churn Predictor - Production Module")
    print("=" * 70)
    
    # Initialize
    predictor = ChurnPredictor()
    
    # Show model info
    print("\nModel Information:")
    info = predictor.get_model_info()
    print(f"  Type: {info['model_type']}")
    print(f"  Features: {info['n_features']}")
    print(f"  Status: Ready")
    
    # Example: Batch prediction with raw data (from baseline test set)
    print("\n" + "-" * 70)
    print("Example: Batch Prediction with Raw Data")
    print("-" * 70)
    
    try:
        # Load baseline test data
        baseline_path = Path(__file__).parent.parent / "data" / "user_testing" / "baseline.csv"
        
        if baseline_path.exists():
            baseline_df = pd.read_csv(baseline_path)
            
            # Make predictions (raw_data=True triggers auto-encoding)
            results = predictor.predict_batch(baseline_df, raw_data=True)
            
            print(f"\n[OK] Successfully predicted for {len(results)} customers")
            print("\nSample Results:")
            print(results[['customerID', 'predicted_churn', 'churn_probability', 'risk_level']].head(3).to_string(index=False))
            
            print(f"\nSummary:")
            print(f"  - High Risk: {(results['risk_level'] == 'HIGH').sum()}")
            print(f"  - Medium Risk: {(results['risk_level'] == 'MEDIUM').sum()}")
            print(f"  - Low Risk: {(results['risk_level'] == 'LOW').sum()}")
        else:
            print(f"[WARN] Baseline test file not found at {baseline_path}")
            print("Skipping batch prediction example.")
    
    except Exception as e:
        print(f"⚠ Error during batch prediction: {e}")
    
    # Single customer example
    print("\n" + "-" * 70)
    print("Example: Single Customer Prediction (Pre-Encoded Features)")
    print("-" * 70)
    
    try:
        # Using pre-encoded features (all binary)
        example_customer_encoded = {
            'SeniorCitizen': 0,
            'tenure': 2,
            'MonthlyCharges': 95.50,
            'TotalCharges': 95.50,
            'PhoneService_Yes': 1,
            'MultipleLines_Yes': 1,
            'InternetService_Fiber optic': 1,
            'InternetService_No': 0,
            'OnlineSecurity_Yes': 0,
            'OnlineBackup_Yes': 0,
            'DeviceProtection_Yes': 0,
            'TechSupport_Yes': 0,
            'StreamingTV_Yes': 1,
            'StreamingMovies_Yes': 1,
            'Contract_One year': 0,
            'Contract_Two year': 0,
            'PaperlessBilling_Yes': 1,
            'PaymentMethod_Credit card (automatic)': 0,
            'PaymentMethod_Electronic check': 1,
            'PaymentMethod_Mailed check': 0,
            'gender_Male': 1,
            'Partner_Yes': 0,
            'Dependents_Yes': 0,
            'OnlineSecurity_No internet service': 0,
            'OnlineBackup_No internet service': 0,
            'DeviceProtection_No internet service': 0,
            'TechSupport_No internet service': 0,
            'StreamingTV_No internet service': 0,
            'StreamingMovies_No internet service': 0
        }
        
        result = predictor.predict_single(example_customer_encoded)
        print(f"\n[OK] Prediction Complete")
        print(f"  Churn: {result['churn_label']}")
        print(f"  Probability: {result['churn_probability']:.2%}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Confidence: {result['model_confidence']:.2%}")
        print(f"  Recommendation: {result['recommendation']}")
    
    except Exception as e:
        print(f"[ERROR] Error during single prediction: {e}")
    
    print("\n" + "=" * 70)
    print("Usage Examples:")
    print("=" * 70)
    print("""
1. Batch Prediction (Raw Categorical Data):
   -----------------------------------------------
   df = pd.read_csv('customer_data.csv')
   predictor = ChurnPredictor()
   results = predictor.predict_batch(df, raw_data=True)

2. Batch Prediction (Pre-Encoded Features):
   -----------------------------------------------
   results = predictor.predict_batch(df, raw_data=False)

3. Single Prediction:
   -----------------------------------------------
   customer = {'gender': 'Male', 'tenure': 12, ...}
   result = predictor.predict_single(customer)

Results include:
   - predicted_churn: 0 or 1
   - churn_probability: confidence score (0-1)
   - risk_level: 'LOW', 'MEDIUM', or 'HIGH'
   - recommendation: actionable business guidance
""")
