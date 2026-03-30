"""
Comprehensive Data Drift Detection Module

Detect distribution shifts between training and new data using multiple statistical tests:
- Z-Score Test: Detects mean shift in numeric features
- Kolmogorov-Smirnov (KS) Test: Tests if distributions match
- Population Stability Index (PSI): Measures magnitude of distribution change
- Chi-Square Test: Tests if categorical proportions differ
- Proportion Shift: Detects category distribution changes

Usage:
    from drift_detector import DriftDetector
    import pandas as pd
    
    # Initialize (auto-loads training stats)
    detector = DriftDetector()
    
    # Check new data batch
    new_data = pd.read_csv('new_batch.csv')
    report = detector.detect_drift(new_data)
    
    # Print formatted report
    detector.print_report(report)
    
    # Export report to JSON
    detector.export_report(report, 'drift_report_20260324.json')
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json
from scipy import stats
from datetime import datetime


class DriftDetector:
    """
    Comprehensive drift detector using multiple statistical tests.
    
    Supports:
    - Numeric features: Z-score, KS test, PSI
    - Categorical features: Proportion shift, Chi-square, PSI
    - Overall drift status with severity levels (LOW, MEDIUM, HIGH)
    
    Thresholds adjusted for small sample sizes (10-100 rows).
    For production with >1000 samples, consider tighter thresholds.
    """
    
    # Numeric feature thresholds
    NUMERIC_THRESHOLDS = {
        'z_score_warning': 1.5,      # > 1.5 std from training mean
        'z_score_critical': 2.5,     # > 2.5 std from training mean
        'ks_warning': 0.3,           # KS statistic (relaxed for small samples)
        'ks_critical': 0.5,
        'psi_warning': 0.1,          # PSI warning threshold
        'psi_critical': 0.25         # PSI critical threshold
    }

    # Effect-size guard rails to avoid p-value-only false positives
    EFFECT_SIZE_THRESHOLDS = {
        # If max proportion shift is below this, chi-square does not escalate severity.
        # This keeps large-but-representative baseline batches from being over-flagged.
        'chi2_min_shift_for_escalation': 0.10
    }
    
    # Categorical feature thresholds
    CATEGORICAL_THRESHOLDS = {
        'proportion_warning': 0.20,   # 20% shift in proportion
        'proportion_critical': 0.35,  # 35% shift in proportion
        'chi2_p_value': 0.01,         # Chi-square significance level
        'psi_warning': 0.1,           # PSI warning threshold
        'psi_critical': 0.25          # PSI critical threshold
    }
    
    def __init__(self, training_stats: Dict = None, stats_path: str = None):
        """
        Initialize drift detector with training statistics.
        
        Args:
            training_stats: Dict with training statistics
            stats_path: Path to JSON file with training statistics
        """
        if training_stats is None and stats_path is None:
            # Try default path
            default_path = Path(__file__).parent.parent / "data" / "processed" / "training_reference_stats.json"
            if default_path.exists():
                stats_path = str(default_path)
            else:
                raise ValueError("Must provide training_stats or stats_path")
        
        if stats_path:
            with open(stats_path, 'r') as f:
                training_stats = json.load(f)
        
        self.training_stats = training_stats
        self.numeric_features = list(training_stats['numeric_features'].keys())
        self.categorical_features = list(training_stats['categorical_features'].keys())
        self.metadata = training_stats.get('metadata', {})
    
    # ========================================================================
    # PSI (Population Stability Index) Calculation
    # ========================================================================
    
    @staticmethod
    def calculate_psi_numeric(train_values: np.ndarray, new_values: np.ndarray, 
                              n_bins: int = 10) -> float:
        """
        Calculate PSI (Population Stability Index) for numeric features.
        
        PSI measures the shift in distribution. Higher values = more drift.
        - PSI < 0.1: No significant population change
        - PSI 0.1-0.25: Small population change (warning)
        - PSI > 0.25: Significant population change (critical)
        
        Args:
            train_values: Training feature values
            new_values: New feature values
            n_bins: Number of bins for distribution discretization
        
        Returns:
            PSI value (float)
        """
        # Create bins based on training data quantiles
        min_val = min(train_values.min(), new_values.min())
        max_val = max(train_values.max(), new_values.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        # Digitize both distributions
        train_binned = np.digitize(train_values, bins)
        new_binned = np.digitize(new_values, bins)
        
        # Calculate proportions
        train_prop = np.bincount(train_binned, minlength=n_bins+2) / len(train_values)
        new_prop = np.bincount(new_binned, minlength=n_bins+2) / len(new_values)
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        train_prop = np.where(train_prop == 0, epsilon, train_prop)
        new_prop = np.where(new_prop == 0, epsilon, new_prop)
        
        # Calculate PSI
        psi = np.sum(new_prop * np.log(new_prop / train_prop))
        return float(psi)
    
    @staticmethod
    def calculate_psi_categorical(train_counts: Dict[str, float], 
                                  new_counts: Dict[str, float]) -> float:
        """
        Calculate PSI for categorical features.
        
        Args:
            train_counts: Dictionary of category proportions in training
            new_counts: Dictionary of category proportions in new data
        
        Returns:
            PSI value (float)
        """
        epsilon = 1e-10
        psi = 0.0
        
        # Get all categories from both distributions
        all_categories = set(train_counts.keys()) | set(new_counts.keys())
        
        for category in all_categories:
            train_prop = train_counts.get(category, 0) + epsilon
            new_prop = new_counts.get(category, 0) + epsilon
            
            psi += new_prop * np.log(new_prop / train_prop)
        
        return float(psi)
    
    # ========================================================================
    # Numeric Drift Detection
    # ========================================================================
    
    def detect_numeric_drift(self, new_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Detect drift in numeric features using multiple tests:
        1. Z-score: Mean shift detection
        2. KS Test: Distribution matching
        3. PSI: Population Stability Index
        
        Args:
            new_data: DataFrame with new data to check
        
        Returns:
            Dict with drift info per numeric feature
        """
        results = {}
        
        for feature in self.numeric_features:
            if feature not in new_data.columns:
                results[feature] = {'status': 'MISSING', 'severity': 'HIGH'}
                continue
            
            train_stats = self.training_stats['numeric_features'][feature]
            new_values = pd.to_numeric(new_data[feature], errors='coerce').dropna().values
            
            if len(new_values) == 0:
                results[feature] = {'status': 'NO_DATA', 'severity': 'HIGH'}
                continue
            
            # Basic statistics
            new_mean = float(np.mean(new_values))
            new_std = float(np.std(new_values))
            train_mean = train_stats['mean']
            train_std = train_stats['std']
            
            # Test 1: Z-score (mean shift)
            z_score = abs(new_mean - train_mean) / train_std if train_std > 0 else 0
            z_severity = 'LOW'
            if z_score >= self.NUMERIC_THRESHOLDS['z_score_critical']:
                z_severity = 'HIGH'
            elif z_score >= self.NUMERIC_THRESHOLDS['z_score_warning']:
                z_severity = 'MEDIUM'
            
            # Detect binary-like numeric features (0/1 style); KS is noisy and not informative here.
            is_binary_numeric = (
                train_stats.get('min') in (0, 0.0)
                and train_stats.get('max') in (1, 1.0)
            )

            # Test 2: KS Test (distribution matching)
            ks_stat, ks_p_value = stats.kstest(
                new_values, 
                'norm', 
                args=(train_mean, train_std)
            )
            ks_severity = 'LOW'
            if not is_binary_numeric:
                if ks_stat >= self.NUMERIC_THRESHOLDS['ks_critical']:
                    ks_severity = 'HIGH'
                elif ks_stat >= self.NUMERIC_THRESHOLDS['ks_warning']:
                    ks_severity = 'MEDIUM'
            
            # Test 3: PSI (Population Stability Index)
            train_values = np.array(train_stats.get('_sample_values', []))
            if len(train_values) > 0:
                psi = self.calculate_psi_numeric(train_values, new_values)
            else:
                # Fallback: estimate PSI from mean/std
                psi = abs(z_score) * 0.1
            
            psi_severity = 'LOW'
            if psi >= self.NUMERIC_THRESHOLDS['psi_critical']:
                psi_severity = 'HIGH'
            elif psi >= self.NUMERIC_THRESHOLDS['psi_warning']:
                psi_severity = 'MEDIUM'
            
            # Overall severity is the worst of robust tests.
            # KS is only included when we have empirical training samples for direct distribution comparison
            # and the feature is not binary-like.
            severity_order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
            include_ks_in_overall = (len(train_values) > 0) and (not is_binary_numeric)
            severity_inputs = [z_severity, psi_severity]
            if include_ks_in_overall:
                severity_inputs.append(ks_severity)

            severity = max(severity_inputs, key=lambda x: severity_order[x])
            
            results[feature] = {
                'train_mean': round(train_mean, 2),
                'train_std': round(train_std, 2),
                'new_mean': round(new_mean, 2),
                'new_std': round(new_std, 2),
                'mean_shift': round(new_mean - train_mean, 2),
                'z_score': round(z_score, 3),
                'z_severity': z_severity,
                'ks_statistic': round(ks_stat, 3),
                'ks_p_value': round(ks_p_value, 4),
                'ks_severity': ks_severity,
                'ks_included_in_overall': include_ks_in_overall,
                'psi': round(psi, 4),
                'psi_severity': psi_severity,
                'severity': severity,
                'drift_detected': severity != 'LOW'
            }
        
        return results
    
    def detect_categorical_drift(self, new_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Detect drift in categorical features using multiple tests:
        1. Proportion shift: Max change in category proportions
        2. Chi-square test: Statistical test for distribution change
        3. PSI: Population Stability Index
        
        Args:
            new_data: DataFrame with new data to check
        
        Returns:
            Dict with drift info per categorical feature
        """
        results = {}
        
        for feature in self.categorical_features:
            if feature not in new_data.columns:
                results[feature] = {'status': 'MISSING', 'severity': 'HIGH'}
                continue
            
            train_stats = self.training_stats['categorical_features'][feature]
            train_proportions = train_stats['proportions']
            train_values = set(train_stats['unique_values'])
            
            # Calculate new proportions
            new_counts = new_data[feature].value_counts()
            new_proportions = (new_counts / len(new_data)).to_dict()
            new_values = set(new_data[feature].unique())
            
            # Check for new/missing categories
            new_categories = new_values - train_values
            missing_categories = train_values - new_values
            
            # Test 1: Proportion shift
            common_categories = train_values & new_values
            max_shift = 0
            shifts = {}
            
            for cat in common_categories:
                train_prop = train_proportions.get(cat, 0)
                new_prop = new_proportions.get(cat, 0)
                shift = abs(new_prop - train_prop)
                shifts[cat] = round(shift, 3)
                max_shift = max(max_shift, shift)
            
            prop_severity = 'LOW'
            if new_categories:
                prop_severity = 'HIGH'
            elif max_shift >= self.CATEGORICAL_THRESHOLDS['proportion_critical']:
                prop_severity = 'HIGH'
            elif max_shift >= self.CATEGORICAL_THRESHOLDS['proportion_warning']:
                prop_severity = 'MEDIUM'
            
            # Test 2: Chi-square test
            n = len(new_data)
            categories = list(train_proportions.keys())
            observed = [new_counts.get(cat, 0) for cat in categories]
            expected = [train_proportions.get(cat, 0) * n for cat in categories]
            
            chi2_stat, chi2_p_value = None, None
            if all(e > 0 for e in expected) and sum(observed) > 0:
                chi2_stat, chi2_p_value = stats.chisquare(observed, expected)
            
            chi2_severity = 'LOW'
            if chi2_p_value is not None:
                # Require meaningful effect size before allowing p-value to escalate severity.
                # This avoids over-triggering on large batches with tiny but harmless differences.
                if max_shift >= self.EFFECT_SIZE_THRESHOLDS['chi2_min_shift_for_escalation']:
                    if chi2_p_value < self.CATEGORICAL_THRESHOLDS['chi2_p_value']:
                        chi2_severity = 'HIGH'
                    elif chi2_p_value < 0.05:
                        chi2_severity = 'MEDIUM'
            
            # Test 3: PSI (Population Stability Index)
            psi = self.calculate_psi_categorical(train_proportions, new_proportions)
            psi_severity = 'LOW'
            if psi >= self.CATEGORICAL_THRESHOLDS['psi_critical']:
                psi_severity = 'HIGH'
            elif psi >= self.CATEGORICAL_THRESHOLDS['psi_warning']:
                psi_severity = 'MEDIUM'
            
            # Overall severity is the worst of all tests
            severity_order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
            severity = max(
                prop_severity, chi2_severity, psi_severity,
                key=lambda x: severity_order[x]
            )
            
            results[feature] = {
                'train_categories': list(train_values),
                'new_categories': list(new_categories) if new_categories else None,
                'missing_categories': list(missing_categories) if missing_categories else None,
                'max_proportion_shift': round(max_shift, 3),
                'proportion_shifts': shifts,
                'proportion_severity': prop_severity,
                'chi2_statistic': round(chi2_stat, 2) if chi2_stat else None,
                'chi2_p_value': round(chi2_p_value, 4) if chi2_p_value else None,
                'chi2_severity': chi2_severity,
                'psi': round(psi, 4),
                'psi_severity': psi_severity,
                'severity': severity,
                'drift_detected': severity != 'LOW'
            }
        
        return results
    
    def detect_drift(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run full drift detection on new data.
        
        Args:
            new_data: DataFrame with new data to check
        
        Returns:
            Comprehensive drift report with:
                - timestamp: Report generation time
                - overall_status: 'OK', 'WARNING', or 'CRITICAL'
                - summary: counts of drifted features by severity
                - numeric_features: detailed drift info per numeric feature
                - categorical_features: detailed drift info per categorical feature
        """
        numeric_drift = self.detect_numeric_drift(new_data)
        categorical_drift = self.detect_categorical_drift(new_data)
        
        # Count drifted features
        numeric_drifted = sum(1 for v in numeric_drift.values() if v.get('drift_detected', False))
        categorical_drifted = sum(1 for v in categorical_drift.values() if v.get('drift_detected', False))
        
        # Count by severity
        all_results = list(numeric_drift.values()) + list(categorical_drift.values())
        high_count = sum(1 for v in all_results if v.get('severity') == 'HIGH')
        medium_count = sum(1 for v in all_results if v.get('severity') == 'MEDIUM')
        
        # Determine overall status
        if high_count > 0:
            overall_status = 'CRITICAL'
        elif medium_count > 0:
            overall_status = 'WARNING'
        else:
            overall_status = 'OK'
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'summary': {
                'total_features': len(self.numeric_features) + len(self.categorical_features),
                'numeric_drifted': numeric_drifted,
                'categorical_drifted': categorical_drifted,
                'high_severity': high_count,
                'medium_severity': medium_count,
                'samples_analyzed': len(new_data)
            },
            'numeric_features': numeric_drift,
            'categorical_features': categorical_drift
        }
    
    def print_report(self, report: Dict[str, Any]):
        """
        Print formatted drift report with all statistical test results.
        
        Args:
            report: Drift report from detect_drift()
        """
        status_icons = {'OK': '✓', 'WARNING': '⚠', 'CRITICAL': '✗'}
        severity_icons = {'LOW': ' ', 'MEDIUM': '*', 'HIGH': '!'}
        
        print("\n" + "=" * 100)
        print(f"DRIFT DETECTION REPORT - {status_icons.get(report['overall_status'], '?')} {report['overall_status']}")
        print(f"Timestamp: {report.get('timestamp', 'N/A')}")
        print("=" * 100)
        
        summary = report['summary']
        print(f"\nSummary:")
        print(f"  Samples analyzed: {summary['samples_analyzed']}")
        print(f"  Total features: {summary['total_features']}")
        print(f"  Features drifted: {summary['numeric_drifted'] + summary['categorical_drifted']}")
        print(f"    - HIGH severity: {summary['high_severity']}")
        print(f"    - MEDIUM severity: {summary['medium_severity']}")
        
        # Numeric features
        print("\n" + "-" * 100)
        print("NUMERIC FEATURES (Z-Score + KS Test + PSI)")
        print("-" * 100)
        print(f"{'':1} {'Feature':<15} {'Z-Score':>8} {'KS Stat':>8} {'KS p-val':>10} {'PSI':>8} {'Severity':>12}")
        print("-" * 100)
        
        for feature, info in report['numeric_features'].items():
            if 'train_mean' in info:
                icon = severity_icons[info['severity']]
                z_score = info.get('z_score', '--')
                ks_stat = info.get('ks_statistic', '--')
                ks_p = info.get('ks_p_value', '--')
                psi = info.get('psi', '--')
                
                z_str = f"{z_score:.3f}" if isinstance(z_score, (int, float)) else z_score
                ks_stat_str = f"{ks_stat:.3f}" if isinstance(ks_stat, (int, float)) else ks_stat
                ks_p_str = f"{ks_p:.4f}" if isinstance(ks_p, float) else ks_p
                psi_str = f"{psi:.4f}" if isinstance(psi, float) else psi
                
                print(f"{icon} {feature:<15} {z_str:>8} {ks_stat_str:>8} {ks_p_str:>10} {psi_str:>8} {info['severity']:>12}")
            else:
                print(f"! {feature:<15} {'--':>8} {'--':>8} {'--':>10} {'--':>8} {info['severity']:>12}")
        
        # Categorical features
        print("\n" + "-" * 100)
        print("CATEGORICAL FEATURES (Proportion + Chi-Square + PSI)")
        print("-" * 100)
        print(f"{'':1} {'Feature':<15} {'Max Shift':>10} {'Chi2 Stat':>9} {'Chi2 p-val':>11} {'PSI':>8} {'Severity':>12}")
        print("-" * 100)
        
        for feature, info in report['categorical_features'].items():
            if 'max_proportion_shift' in info:
                icon = severity_icons[info['severity']]
                max_shift = info.get('max_proportion_shift', '--')
                chi2_stat = info.get('chi2_statistic', '--')
                chi2_p = info.get('chi2_p_value', '--')
                psi = info.get('psi', '--')
                
                max_shift_str = f"{max_shift:.3f}" if isinstance(max_shift, (int, float)) else max_shift
                chi2_stat_str = f"{chi2_stat:.2f}" if isinstance(chi2_stat, (int, float)) else chi2_stat
                chi2_p_str = f"{chi2_p:.4f}" if isinstance(chi2_p, float) else chi2_p
                psi_str = f"{psi:.4f}" if isinstance(psi, float) else psi
                
                print(f"{icon} {feature:<15} {max_shift_str:>10} {chi2_stat_str:>9} {chi2_p_str:>11} {psi_str:>8} {info['severity']:>12}")
            else:
                print(f"! {feature:<15} {'--':>10} {'--':>9} {'--':>11} {'--':>8} {info['severity']:>12}")
        
        print("\n" + "=" * 100)
        print("TEST INTERPRETATION:")
        print(f"  Z-Score: Detects mean shift in units of standard deviation")
        print(f"  KS Test: p < 0.05 suggests distributions differ significantly")
        print(f"  PSI:     < 0.1 (no change), 0.1-0.25 (warning), > 0.25 (critical)")
        print(f"  Chi-Square: p < 0.01 suggests category proportions shifted significantly")
        print("-" * 100)
        
        if report['overall_status'] == 'CRITICAL':
            print("⚡ ALERT: Significant drift detected!")
            print("  Recommended actions:")
            print("   1. Investigate root cause of data shift")
            print("   2. Assess impact on model performance")
            print("   3. Consider model retraining with recent data")
        elif report['overall_status'] == 'WARNING':
            print("⚠️  WARNING: Moderate drift detected.")
            print("  Recommended actions:")
            print("   1. Monitor drift trends over next few batches")
            print("   2. Review affected features for data quality issues")
            print("   3. Plan model performance evaluation")
        else:
            print("✓ OK: No significant drift detected. Continue normal operations.")
        
        print("=" * 100)
    
    def get_drifted_features(self, report: Dict[str, Any], min_severity: str = 'MEDIUM') -> List[str]:
        """
        Get list of features with drift at or above specified severity.
        
        Args:
            report: Drift report from detect_drift()
            min_severity: Minimum severity level ('LOW', 'MEDIUM', 'HIGH')
        
        Returns:
            List of feature names with drift
        """
        severity_order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        min_level = severity_order.get(min_severity, 1)
        
        drifted = []
        
        for feature, info in report['numeric_features'].items():
            if severity_order.get(info.get('severity', 'LOW'), 0) >= min_level:
                drifted.append(feature)
        
        for feature, info in report['categorical_features'].items():
            if severity_order.get(info.get('severity', 'LOW'), 0) >= min_level:
                drifted.append(feature)
        
        return drifted
    
    def export_report(self, report: Dict[str, Any], filepath: str = None) -> str:
        """
        Export drift report to JSON file.
        
        Args:
            report: Drift report from detect_drift()
            filepath: Path to save report (optional, defaults to timestamped file)
        
        Returns:
            Path where report was saved
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            history_dir = Path(__file__).parent.parent / 'data' / 'drift_history'
            history_dir.mkdir(parents=True, exist_ok=True)
            filepath = history_dir / f'drift_report_{timestamp}.json'
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(filepath)
    
    def get_report_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a concise summary of the drift report for quick reference.
        
        Args:
            report: Drift report from detect_drift()
        
        Returns:
            Summary dict with top-level statistics
        """
        summary = report['summary'].copy()
        summary['timestamp'] = report.get('timestamp', 'N/A')
        summary['overall_status'] = report['overall_status']
        
        # Add high-severity features
        high_severity_features = self.get_drifted_features(report, 'HIGH')
        medium_severity_features = self.get_drifted_features(report, 'MEDIUM')
        
        summary['high_severity_features'] = high_severity_features
        summary['medium_severity_features'] = medium_severity_features
        
        return summary



# ============================================================================
# Convenience functions
# ============================================================================

def check_drift(data: pd.DataFrame, stats_path: str = None) -> Dict[str, Any]:
    """
    Quick drift check on new data.
    
    Args:
        data: DataFrame with new data
        stats_path: Path to training stats JSON (optional)
    
    Returns:
        Drift report
    """
    detector = DriftDetector(stats_path=stats_path)
    return detector.detect_drift(data)


if __name__ == "__main__":
    # Demo usage
    print("=" * 100)
    print("ENHANCED DRIFT DETECTION MODULE - DEMO")
    print("=" * 100)
    
    # Initialize
    detector = DriftDetector()
    print(f"\nDetector initialized with training statistics")
    print(f"  Numeric features: {len(detector.numeric_features)}")
    print(f"  Categorical features: {len(detector.categorical_features)}")
    
    # Test with baseline data
    baseline_path = Path(__file__).parent.parent / "data" / "user_testing" / "baseline.csv"
    if baseline_path.exists():
        print("\n" + "√" * 50)
        print("TEST 1: Baseline Data (Expected: Minimal Drift)")
        print("√" * 50)
        baseline_df = pd.read_csv(baseline_path)
        report = detector.detect_drift(baseline_df)
        detector.print_report(report)
        
        # Export report
        saved_path = detector.export_report(report)
        print(f"\n✓ Report saved to: {saved_path}")
        
        # Get summary
        summary = detector.get_report_summary(report)
        print(f"\nQuick Summary: {report['overall_status']} | {summary['high_severity']} HIGH, {summary['medium_severity']} MEDIUM")
    else:
        print(f"\n✗ Baseline data not found at {baseline_path}")
    
    # Test with drifted data
    drifted_path = Path(__file__).parent.parent / "data" / "user_testing" / "drifted_data.csv"
    if drifted_path.exists():
        print("\n" + "√" * 50)
        print("TEST 2: Drifted Data (Expected: Significant Drift)")
        print("√" * 50)
        drifted_df = pd.read_csv(drifted_path)
        report = detector.detect_drift(drifted_df)
        detector.print_report(report)
        
        # Export report
        saved_path = detector.export_report(report)
        print(f"\n✓ Report saved to: {saved_path}")
        
        # Get summary
        summary = detector.get_report_summary(report)
        print(f"\nQuick Summary: {report['overall_status']} | {summary['high_severity']} HIGH, {summary['medium_severity']} MEDIUM")
        print(f"High severity features: {summary['high_severity_features']}")
    else:
        print(f"\n✗ Drifted data not found at {drifted_path}")
    
    print("\n" + "=" * 100)
