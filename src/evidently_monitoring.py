"""
Evidently AI Monitoring Integration

Provides comprehensive data and prediction drift monitoring using Evidently.
Complements custom drift detector with professional visualization and reporting.

Usage:
    from evidently_monitoring import EvidentlyMonitoring
    import pandas as pd
    
    # Initialize with training data
    monitoring = EvidentlyMonitoring(reference_data, target_column='churn')
    
    # Generate comprehensive report
    report = monitoring.generate_dataset_drift_report(new_data)
    report.save_html('drift_report.html')
    
    # Get structured metrics
    metrics = monitoring.get_drift_metrics(new_data)
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.metrics import (
        DataDriftTable, DatasetDriftMetric, ColumnDriftMetric,
        ColumnValueDriftMetric, ColumnCorrelationsDriftMetric
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("⚠️  Evidently AI not installed. Run: pip install evidently")


class EvidentlyMonitoring:
    """
    Wrapper for Evidently AI monitoring capabilities.
    
    Provides:
    - Dataset drift detection (numeric + categorical)
    - Target drift detection  
    - Feature correlation analysis
    - Detailed HTML reports
    - Structured metrics export
    """
    
    def __init__(self, reference_data: pd.DataFrame, 
                 target_column: Optional[str] = None,
                 categorical_features: Optional[List[str]] = None,
                 numerical_features: Optional[List[str]] = None):
        """
        Initialize Evidently monitoring.
        
        Args:
            reference_data: Training/baseline data for comparison
            target_column: Name of target column (for target drift)
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
        """
        if not EVIDENTLY_AVAILABLE:
            raise ImportError("Evidently AI is not installed. Install with: pip install evidently")
        
        self.reference_data = reference_data
        self.target_column = target_column
        
        # Auto-detect feature types if not provided
        if categorical_features is None or numerical_features is None:
            numeric_dtypes = ['int64', 'float64']
            categorical_dtypes = ['object', 'category']
            
            if numerical_features is None:
                numerical_features = reference_data.select_dtypes(
                    include=numeric_dtypes
                ).columns.tolist()
                if target_column in numerical_features:
                    numerical_features.remove(target_column)
            
            if categorical_features is None:
                categorical_features = reference_data.select_dtypes(
                    include=categorical_dtypes
                ).columns.tolist()
                if target_column in categorical_features:
                    categorical_features.remove(target_column)
        
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.feature_list = numerical_features + categorical_features
        
        # Setup column mapping
        self.column_mapping = ColumnMapping(
            target=target_column,
            numerical_features=numerical_features,
            categorical_features=categorical_features
        )
    
    def generate_dataset_drift_report(self, current_data: pd.DataFrame,
                                     report_name: str = None) -> Report:
        """
        Generate comprehensive dataset drift report using Evidently.
        
        Args:
            current_data: New data batch to check for drift
            report_name: Optional name for the report
        
        Returns:
            Evidently Report object
        """
        if not EVIDENTLY_AVAILABLE:
            raise ImportError("Evidently not available")
        
        report = Report(
            metrics=[
                DataDriftPreset(),  # Full dataset drift analysis
                DatasetDriftMetric(),
                DataDriftTable(),
            ]
        )
        
        try:
            report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            return report
        except Exception as e:
            raise RuntimeError(f"Failed to generate Evidently report: {str(e)}")
    
    def generate_target_drift_report(self, current_data: pd.DataFrame) -> Report:
        """
        Generate target drift report (if target column exists).
        
        Args:
            current_data: New data with target values
        
        Returns:
            Evidently Report object
        """
        if not EVIDENTLY_AVAILABLE:
            raise ImportError("Evidently not available")
        
        if self.target_column is None:
            raise ValueError("Target column not specified")
        
        report = Report(metrics=[TargetDriftPreset()])
        
        try:
            report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            return report
        except Exception as e:
            raise RuntimeError(f"Failed to generate target drift report: {str(e)}")
    
    def save_report_html(self, report: Report, filepath: str = None) -> str:
        """
        Save Evidently report as HTML.
        
        Args:
            report: Evidently Report object
            filepath: Path to save (default: timestamped file)
        
        Returns:
            Path where report was saved
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            history_dir = Path(__file__).parent.parent / 'data' / 'drift_history'
            history_dir.mkdir(parents=True, exist_ok=True)
            filepath = history_dir / f'evidently_drift_report_{timestamp}.html'
        
        report.save_html(str(filepath))
        return str(filepath)
    
    def get_drift_metrics(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract quantitative drift metrics from Evidently analysis.
        
        Args:
            current_data: New data batch
        
        Returns:
            Dictionary with drift metrics for each feature
        """
        report = self.generate_dataset_drift_report(current_data)
        
        metrics_dict = {
            'timestamp': datetime.now().isoformat(),
            'samples': len(current_data),
            'features': {}
        }
        
        # Extract metrics from report
        try:
            dashboard_data = report.as_dict()
            
            # Parse drift results
            for metric in dashboard_data.get('metrics', []):
                metric_name = metric.get('metric', {}).get('name', '')
                result = metric.get('result', {})
                
                if 'drift_by_columns' in result:
                    for feature_name, feature_data in result['drift_by_columns'].items():
                        metrics_dict['features'][feature_name] = {
                            'drift_detected': feature_data.get('drift_detected', False),
                            'result': feature_data.get('stattest_result', {}),
                            'type': self._get_feature_type(feature_name)
                        }
        except Exception as e:
            print(f"⚠️  Warning: Could not extract detailed metrics: {str(e)}")
        
        return metrics_dict
    
    def _get_feature_type(self, feature_name: str) -> str:
        """Helper to determine feature type."""
        if feature_name in self.numerical_features:
            return 'numeric'
        elif feature_name in self.categorical_features:
            return 'categorical'
        return 'unknown'
    
    def export_metrics_json(self, current_data: pd.DataFrame,
                           filepath: str = None) -> str:
        """
        Export drift metrics as JSON.
        
        Args:
            current_data: New data batch
            filepath: Path to save (default: timestamped file)
        
        Returns:
            Path where metrics were saved
        """
        metrics = self.get_drift_metrics(current_data)
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            history_dir = Path(__file__).parent.parent / 'data' / 'drift_history'
            history_dir.mkdir(parents=True, exist_ok=True)
            filepath = history_dir / f'evidently_metrics_{timestamp}.json'
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        return str(filepath)
    
    def compare_distributions(self, current_data: pd.DataFrame,
                            feature: str) -> Dict[str, Any]:
        """
        Compare distribution of a specific feature between reference and current.
        
        Args:
            current_data: New data batch
            feature: Feature name to analyze
        
        Returns:
            Comparison statistics
        """
        if feature not in self.feature_list:
            raise ValueError(f"Feature '{feature}' not found in feature list")
        
        ref_data = self.reference_data[feature]
        cur_data = current_data[feature]
        
        if feature in self.numerical_features:
            return self._compare_numeric_dist(ref_data, cur_data, feature)
        else:
            return self._compare_categorical_dist(ref_data, cur_data, feature)
    
    def _compare_numeric_dist(self, ref_data: pd.Series,
                             cur_data: pd.Series,
                             feature: str) -> Dict[str, Any]:
        """Compare numeric distributions."""
        ref_data = pd.to_numeric(ref_data, errors='coerce').dropna()
        cur_data = pd.to_numeric(cur_data, errors='coerce').dropna()
        
        return {
            'feature': feature,
            'type': 'numeric',
            'reference': {
                'mean': float(ref_data.mean()),
                'std': float(ref_data.std()),
                'min': float(ref_data.min()),
                'max': float(ref_data.max()),
                'median': float(ref_data.median()),
                'count': len(ref_data)
            },
            'current': {
                'mean': float(cur_data.mean()),
                'std': float(cur_data.std()),
                'min': float(cur_data.min()),
                'max': float(cur_data.max()),
                'median': float(cur_data.median()),
                'count': len(cur_data)
            },
            'changes': {
                'mean_difference': float(cur_data.mean() - ref_data.mean()),
                'mean_pct_change': float((cur_data.mean() - ref_data.mean()) / ref_data.mean() * 100) if ref_data.mean() != 0 else None
            }
        }
    
    def _compare_categorical_dist(self, ref_data: pd.Series,
                                  cur_data: pd.Series,
                                  feature: str) -> Dict[str, Any]:
        """Compare categorical distributions."""
        ref_counts = ref_data.value_counts(normalize=True)
        cur_counts = cur_data.value_counts(normalize=True)
        
        return {
            'feature': feature,
            'type': 'categorical',
            'reference': {
                'value_counts': ref_counts.to_dict(),
                'unique_count': ref_data.nunique()
            },
            'current': {
                'value_counts': cur_counts.to_dict(),
                'unique_count': cur_data.nunique()
            },
            'new_categories': list(set(cur_data.unique()) - set(ref_data.unique())),
            'removed_categories': list(set(ref_data.unique()) - set(cur_data.unique()))
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_evidently_report(reference_data: pd.DataFrame,
                             current_data: pd.DataFrame,
                             target_column: Optional[str] = None,
                             output_html: str = None) -> str:
    """
    Quick function to generate Evidently drift report.
    
    Args:
        reference_data: Training/baseline data
        current_data: New data to check
        target_column: Target column name (optional)
        output_html: Path to save HTML report
    
    Returns:
        Path to saved HTML report
    """
    monitoring = EvidentlyMonitoring(reference_data, target_column)
    report = monitoring.generate_dataset_drift_report(current_data)
    
    if output_html is None:
        output_html = monitoring.save_report_html(report)
    else:
        report.save_html(output_html)
    
    return output_html


if __name__ == "__main__":
    print("Evidently AI Monitoring Module")
    print("Use: from evidently_monitoring import EvidentlyMonitoring")
