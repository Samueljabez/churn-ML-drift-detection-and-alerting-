"""
Interactive Drift Detection Dashboard

Streamlit-based dashboard for monitoring and visualizing data drift over time.
Displays metrics from custom drift detector and alerting reports.

Usage:
    streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
from scipy.stats import gaussian_kde

# Page configuration
st.set_page_config(
    page_title="Drift Detection Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at 15% 20%, rgba(255, 196, 0, 0.15), transparent 28%),
            radial-gradient(circle at 82% 24%, rgba(255, 87, 87, 0.16), transparent 30%),
            radial-gradient(circle at 34% 80%, rgba(0, 191, 166, 0.18), transparent 32%),
            linear-gradient(145deg, #0a1022 0%, #101936 45%, #121d3f 100%);
    }
    h1, h2, h3 {
        letter-spacing: 0.3px;
    }
    .hero-card {
        background: linear-gradient(120deg, rgba(255, 174, 0, 0.24), rgba(0, 201, 255, 0.22));
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 8px 26px rgba(0, 0, 0, 0.25);
    }
    .artifact-card {
        background: linear-gradient(135deg, rgba(21, 30, 61, 0.95), rgba(35, 48, 94, 0.95));
        border-left: 5px solid #00d2ff;
        border-radius: 12px;
        padding: 10px 14px;
        margin: 6px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .status-ok { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-critical { color: #dc3545; font-weight: bold; }
    .window-shell {
        background: linear-gradient(140deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02));
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 18px;
        padding: 14px 16px 20px 16px;
        margin-top: 6px;
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.26);
        animation: slide-in 420ms cubic-bezier(0.2, 0.85, 0.32, 1);
    }
    .window-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
        padding: 6px 10px;
        border-radius: 10px;
        background: rgba(10, 16, 34, 0.48);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .window-dots {
        display: inline-flex;
        gap: 6px;
    }
    .window-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .window-title {
        font-size: 0.9rem;
        opacity: 0.85;
        letter-spacing: 0.2px;
    }
    @keyframes slide-in {
        from {
            opacity: 0;
            transform: translateX(24px) scale(0.992);
        }
        to {
            opacity: 1;
            transform: translateX(0) scale(1);
        }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_drift_history(history_dir: str = None, limit: int = 100) -> List[Dict]:
    """Load drift monitoring history."""
    if history_dir is None:
        history_dir = Path(__file__).parent.parent / 'data' / 'drift_history'
    
    history_dir = Path(history_dir)
    if not history_dir.exists():
        return []
    
    reports = []
    drift_files = sorted(
        history_dir.glob('drift_report_*.json'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for filepath in drift_files[:limit]:
        try:
            with open(filepath, 'r') as f:
                report = json.load(f)
                report['_filepath'] = str(filepath)
                reports.append(report)
        except:
            pass
    
    return reports


def load_latest_report(history_dir: str = None) -> Optional[Dict]:
    """Load the most recent drift report."""
    reports = load_drift_history(history_dir, limit=1)
    return reports[0] if reports else None


def load_latest_combined_report(history_dir: str = None) -> Optional[Dict]:
    """Load most recent combined monitoring report."""
    reports = get_combined_monitoring_history(history_dir, limit=1)
    return reports[0] if reports else None


def get_combined_monitoring_history(history_dir: str = None, limit: int = 100) -> List[Dict]:
    """Load combined monitoring reports."""
    if history_dir is None:
        history_dir = Path(__file__).parent.parent / 'data' / 'drift_history'
    
    history_dir = Path(history_dir)
    if not history_dir.exists():
        return []
    
    reports = []
    combined_files = sorted(
        history_dir.glob('combined_monitoring_report_*.json'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for filepath in combined_files[:limit]:
        try:
            with open(filepath, 'r') as f:
                report = json.load(f)
                report['_filepath'] = str(filepath)
                reports.append(report)
        except:
            pass
    
    return reports


@st.cache_data(ttl=300)
def get_latest_prediction_png(predictions_dir: str = None, patterns: Optional[List[str]] = None) -> Optional[Path]:
    """Load newest prediction-related PNG artifact by pattern list."""
    if predictions_dir is None:
        predictions_dir = Path(__file__).parent.parent / 'data' / 'predictions'

    predictions_path = Path(predictions_dir)
    if not predictions_path.exists():
        return None

    png_patterns = patterns or ['interactive_dashboard_*.png', 'single_prediction_*.png', 'batch_prediction_*.png']
    png_files: List[Path] = []
    for pattern in png_patterns:
        png_files.extend(predictions_path.glob(pattern))

    if not png_files:
        return None

    return max(png_files, key=lambda p: p.stat().st_mtime)


def parse_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """Extract timestamp like YYYYMMDD_HHMMSS from filename."""
    match = re.search(r'(\d{8}_\d{6})', filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
    except ValueError:
        return None


# ============================================================================
# Helper Functions
# ============================================================================

def get_status_color(status: str) -> str:
    """Get color for status."""
    if status == 'CRITICAL':
        return '#dc3545'
    elif status == 'WARNING':
        return '#ffc107'
    else:
        return '#28a745'


def get_status_emoji(status: str) -> str:
    """Get emoji for status."""
    if status == 'CRITICAL':
        return '🚨'
    elif status == 'WARNING':
        return '⚠️'
    else:
        return '✓'


def format_timestamp(ts_str: str) -> str:
    """Format ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return ts_str


def load_visualization_frames() -> Optional[Dict[str, Any]]:
    """Load reference/current frames for drift visualizations."""
    data_root = Path(__file__).parent.parent / 'data'
    user_testing_dir = data_root / 'user_testing'

    # Try latest combined report metadata first.
    latest_combined = load_latest_combined_report()
    reference_path = None
    current_path = None

    if latest_combined is not None:
        input_meta = latest_combined.get('input_metadata', {})
        ref_meta = input_meta.get('reference_data_path')
        cur_meta = input_meta.get('current_data_path')
        if ref_meta:
            ref_candidate = Path(ref_meta)
            if ref_candidate.exists():
                reference_path = ref_candidate
        if cur_meta:
            cur_candidate = Path(cur_meta)
            if cur_candidate.exists():
                current_path = cur_candidate

    # Fallbacks for notebook-driven runs where metadata paths may be missing.
    if reference_path is None:
        for fallback in [user_testing_dir / 'baseline_large.csv', user_testing_dir / 'baseline.csv']:
            if fallback.exists():
                reference_path = fallback
                break

    if current_path is None:
        for fallback in [user_testing_dir / 'drifted_data.csv', user_testing_dir / 'baseline.csv']:
            if fallback.exists():
                current_path = fallback
                break

    if reference_path is None or current_path is None:
        return None

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    numeric_cols = [
        c for c in reference_df.select_dtypes(include=[np.number]).columns
        if c in current_df.columns
    ]

    if not numeric_cols:
        return None

    return {
        'reference_df': reference_df,
        'current_df': current_df,
        'numeric_cols': numeric_cols,
        'reference_path': str(reference_path),
        'current_path': str(current_path),
    }


def get_latest_drift_payload() -> Optional[Dict[str, Any]]:
    """Normalize latest drift payload from either combined or standalone report."""
    latest_combined = load_latest_combined_report()
    if latest_combined is not None:
        detector = latest_combined.get('custom_detector', {})
        return {
            'status': detector.get('overall_status', 'UNKNOWN'),
            'summary': detector.get('summary', {}),
            'numeric_features': detector.get('numeric_features', {}),
            'categorical_features': detector.get('categorical_features', {}),
            'timestamp': latest_combined.get('timestamp', '--'),
            'samples_analyzed': latest_combined.get('run_metadata', {}).get('samples_analyzed', 0),
            'alerting': latest_combined.get('alerting', {}),
        }

    latest = load_latest_report()
    if latest is not None:
        return {
            'status': latest.get('overall_status', 'UNKNOWN'),
            'summary': latest.get('summary', {}),
            'numeric_features': latest.get('numeric_features', {}),
            'categorical_features': latest.get('categorical_features', {}),
            'timestamp': latest.get('timestamp', '--'),
            'samples_analyzed': latest.get('summary', {}).get('samples_analyzed', 0),
            'alerting': {},
        }

    return None


def _severity_score(severity: str) -> float:
    return {'HIGH': 3.0, 'MEDIUM': 2.0, 'LOW': 1.0}.get(str(severity).upper(), 0.0)


def rank_drift_features(payload: Dict[str, Any], top_n: int = 4) -> Dict[str, List[Dict[str, Any]]]:
    """Rank numeric and categorical features by drift significance."""
    ranked_numeric: List[Dict[str, Any]] = []
    ranked_categorical: List[Dict[str, Any]] = []

    for feature, info in payload.get('numeric_features', {}).items():
        psi = float(info.get('psi', 0) or 0)
        z_score = abs(float(info.get('z_score', 0) or 0))
        score = _severity_score(info.get('severity', 'LOW')) * 10 + psi * 5 + z_score
        ranked_numeric.append({'feature': feature, 'score': score, 'info': info})

    for feature, info in payload.get('categorical_features', {}).items():
        psi = float(info.get('psi', 0) or 0)
        max_shift = float(info.get('max_proportion_shift', 0) or 0)
        score = _severity_score(info.get('severity', 'LOW')) * 10 + max_shift * 20 + psi * 5
        ranked_categorical.append({'feature': feature, 'score': score, 'info': info})

    ranked_numeric.sort(key=lambda x: x['score'], reverse=True)
    ranked_categorical.sort(key=lambda x: x['score'], reverse=True)

    return {
        'numeric': ranked_numeric[:top_n],
        'categorical': ranked_categorical[:top_n],
    }


def section_home():
    """Home screen to select data and run drift detection."""
    st.markdown("<h2>🏠 Data Drift & Prediction Home</h2>", unsafe_allow_html=True)
    st.write("Select a batch of customer data to analyze for potential drift against standard training baselines.")
    
    # Locate data files
    data_dir = Path("data")
    testing_dir = data_dir / "user_testing"
    stats_path = data_dir / "processed" / "training_reference_stats.json"
    
    if not testing_dir.exists():
        st.warning(f"Could not find `{testing_dir}` directory.")
        return

    csv_files = [f.name for f in testing_dir.glob("*.csv")]
    if not csv_files:
        st.info(f"No CSV files found in `{testing_dir}`.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_file = st.selectbox("Select CSV Batch for Analysis:", csv_files)
    
    if st.button("Run Drift Detection", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing {selected_file} for Data Drift..."):
            try:
                import sys
                src_path = str(Path(__file__).parent.resolve())
                if src_path not in sys.path:
                    sys.path.append(src_path)
                
                from run_monitoring import DriftMonitor
                
                current_df = pd.read_csv(testing_dir / selected_file)
                
                # Pick baseline reference securely
                ref_file = "baseline_large.csv" if "baseline_large.csv" in csv_files else csv_files[0]
                reference_df = pd.read_csv(testing_dir / ref_file)
                
                # Initialize & Run Monitor
                monitor = DriftMonitor(
                    reference_data=reference_df,
                    training_stats_path=str(stats_path) if stats_path.exists() else None,
                    alerts_enabled=True,
                    alert_on="warning"
                )
                
                # We save it using a unique UI batch name
                report = monitor.run_monitoring(
                    current_data=current_df,
                    batch_name=f"ui_{selected_file.split('.')[0]}_{datetime.now().strftime('%H%M%S')}"
                )
                
                st.success(f"✅ Drift Detection Completed! Result: **{report['combined_verdict']}**")
                st.info("The latest report has been generated. Use the top navigation (▶ or click '📡 Drift Section') to view detailed analysis.")
                
            except Exception as e:
                st.error(f"Error running drift detection: {str(e)}")


def section_drift_command_center():
    """High-level command center shown first in drift area."""
    payload = get_latest_drift_payload()
    if payload is None:
        st.info("No drift detection reports found. Run monitoring first: python src/run_monitoring.py --data <file>")
        return

    status = payload.get('status', 'UNKNOWN')
    summary = payload.get('summary', {})
    alerting = payload.get('alerting', {})
    status_class = 'status-ok'
    if status == 'CRITICAL':
        status_class = 'status-critical'
    elif status == 'WARNING':
        status_class = 'status-warning'

    st.markdown(
        f"""
        <div class="hero-card">
            <h3 style="margin:0;">Drift Detection Command Center</h3>
            <p style="margin:6px 0 0 0;" class="{status_class}">{get_status_emoji(status)} {status}</p>
            <p style="margin:6px 0 0 0; opacity:0.9;">Latest check: {format_timestamp(str(payload.get('timestamp', '--')))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Samples", payload.get('samples_analyzed', 0))
    c2.metric("Total Features", summary.get('total_features', 0))
    c3.metric("High Severity", summary.get('high_severity', 0), delta="urgent")
    c4.metric("Medium Severity", summary.get('medium_severity', 0), delta="monitor")
    c5.metric("Alert", alerting.get('alert_level', 'NONE'))

    if alerting:
        st.markdown(
            f"""
            <div class="artifact-card">
                <b>Alert Triggered:</b> {alerting.get('alert_triggered', False)} &nbsp;|&nbsp;
                <b>Message:</b> {alerting.get('alert_message', 'No alert message')}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# Dashboard Sections
# ============================================================================

def section_overview():
    """Overall status and summary."""
    st.header("📊 Drift Detection Overview")
    
    # Prefer combined report (custom + alerting), fallback to custom-only report.
    latest_combined = load_latest_combined_report()
    latest_report = load_latest_report()

    if latest_combined is None and latest_report is None:
        st.info("No drift detection reports found. Run monitoring first: `python src/run_monitoring.py --data <file>`")
        return

    if latest_combined is not None:
        status = latest_combined.get('custom_detector', {}).get('overall_status', 'UNKNOWN')
        summary = latest_combined.get('custom_detector', {}).get('summary', {})
        timestamp_value = latest_combined.get('timestamp', '--')
        samples_analyzed = latest_combined.get('run_metadata', {}).get('samples_analyzed', 0)
        alerting_info = latest_combined.get('alerting', {})
    else:
        status = latest_report['overall_status']
        summary = latest_report['summary']
        timestamp_value = latest_report.get('timestamp', '--')
        samples_analyzed = latest_report['summary'].get('samples_analyzed', 0)
        alerting_info = {}
    
    # Status cards
    col1, col2, col3, col4, col5 = st.columns(5)

    status_color = get_status_color(status)
    status_emoji = get_status_emoji(status)
    
    with col1:
        st.metric(
            "Overall Status",
            f"{status_emoji} {status}",
            delta=None,
            help="CRITICAL: Significant drift | WARNING: Moderate drift | OK: No drift"
        )
    
    with col2:
        st.metric(
            "Samples Analyzed",
            samples_analyzed,
            help="Number of records in current batch"
        )
    
    with col3:
        high = summary.get('high_severity', 0)
        st.metric(
            "High Severity Features",
            high,
            delta="features drifted",
            delta_color="inverse"
        )
    
    with col4:
        medium = summary.get('medium_severity', 0)
        st.metric(
            "Medium Severity Features",
            medium,
            delta="features drifted"
        )

    with col5:
        alert_level = alerting_info.get('alert_level', 'N/A')
        st.metric(
            "Alert Level",
            alert_level,
            help="NONE, WARNING, or CRITICAL"
        )
    
    # Timestamp
    st.caption(f"Last check: {format_timestamp(timestamp_value)}")

    if latest_combined is not None and alerting_info:
        st.info(
            f"Alert triggered: {alerting_info.get('alert_triggered', False)} | "
            f"Message: {alerting_info.get('alert_message', 'No alert message')}"
        )
    
    # Summary table
    st.subheader("Summary Statistics")
    summary_data = {
        'Metric': [
            'Total Features',
            'Features with Drift',
            'Numeric Features Drifted',
            'Categorical Features Drifted',
            'High Severity',
            'Medium Severity'
        ],
        'Count': [
            summary.get('total_features', 0),
            summary.get('numeric_drifted', 0) + summary.get('categorical_drifted', 0),
            summary.get('numeric_drifted', 0),
            summary.get('categorical_drifted', 0),
            summary.get('high_severity', 0),
            summary.get('medium_severity', 0)
        ]
    }
    
    st.dataframe(
        pd.DataFrame(summary_data),
        width='stretch',
        hide_index=True
    )


def section_feature_details():
    """Detailed feature-level drift metrics."""
    st.header("🔍 Feature-Level Drift Analysis")
    
    latest_report = load_latest_report()
    
    if latest_report is None:
        st.info("No drift reports available")
        return
    
    # Tabs for numeric and categorical features
    tab1, tab2 = st.tabs(["Numeric Features", "Categorical Features"])
    
    with tab1:
        st.subheader("Numeric Features (Z-Score, KS Test, PSI)")
        
        if not latest_report['numeric_features']:
            st.info("No numeric features found")
        else:
            # Prepare data
            numeric_data = []
            for feature_name, feature_info in latest_report['numeric_features'].items():
                if 'train_mean' in feature_info:
                    numeric_data.append({
                        'Feature': feature_name,
                        'Z-Score': feature_info.get('z_score', '--'),
                        'KS Statistic': feature_info.get('ks_statistic', '--'),
                        'KS p-value': feature_info.get('ks_p_value', '--'),
                        'PSI': feature_info.get('psi', '--'),
                        'Severity': feature_info.get('severity', '--'),
                        'Drift': '✓' if feature_info.get('drift_detected') else '✗'
                    })
            
            if numeric_data:
                df_numeric = pd.DataFrame(numeric_data)
                
                # Color code severity
                def highlight_severity(val):
                    if val == 'HIGH':
                        return 'background-color: #ffe6e6'
                    elif val == 'MEDIUM':
                        return 'background-color: #ffffcc'
                    else:
                        return 'background-color: #e6ffe6'
                
                styled_df = df_numeric.style.map(
                    lambda x: 'background-color: #ffe6e6' if x == 'HIGH'
                              else ('background-color: #ffffcc' if x == 'MEDIUM' else ''),
                    subset=['Severity']
                )
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Z-Score Distribution")
                    fig = px.bar(
                        df_numeric,
                        x='Feature',
                        y='Z-Score',
                        color='Severity',
                        color_discrete_map={'HIGH': '#dc3545', 'MEDIUM': '#ffc107', 'LOW': '#28a745'},
                        title='Mean Shift (Z-Score)',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("PSI Distribution")
                    fig = px.bar(
                        df_numeric,
                        x='Feature',
                        y='PSI',
                        color='Severity',
                        color_discrete_map={'HIGH': '#dc3545', 'MEDIUM': '#ffc107', 'LOW': '#28a745'},
                        title='Population Stability Index',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Categorical Features (Chi-Square, Proportion Shift, PSI)")
        
        if not latest_report['categorical_features']:
            st.info("No categorical features found")
        else:
            # Prepare data
            categorical_data = []
            for feature_name, feature_info in latest_report['categorical_features'].items():
                if 'max_proportion_shift' in feature_info:
                    categorical_data.append({
                        'Feature': feature_name,
                        'Max Shift': feature_info.get('max_proportion_shift', '--'),
                        'Chi2 Stat': feature_info.get('chi2_statistic', '--'),
                        'Chi2 p-value': feature_info.get('chi2_p_value', '--'),
                        'PSI': feature_info.get('psi', '--'),
                        'Severity': feature_info.get('severity', '--'),
                        'New Categories': len(feature_info.get('new_categories', [])) if feature_info.get('new_categories') else 0,
                        'Drift': '✓' if feature_info.get('drift_detected') else '✗'
                    })
            
            if categorical_data:
                df_categorical = pd.DataFrame(categorical_data)
                
                st.dataframe(df_categorical, use_container_width=True, hide_index=True)
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Proportion Shift Distribution")
                    fig = px.bar(
                        df_categorical,
                        x='Feature',
                        y='Max Shift',
                        color='Severity',
                        color_discrete_map={'HIGH': '#dc3545', 'MEDIUM': '#ffc107', 'LOW': '#28a745'},
                        title='Maximum Category Proportion Change',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("PSI Distribution")
                    fig = px.bar(
                        df_categorical,
                        x='Feature',
                        y='PSI',
                        color='Severity',
                        color_discrete_map={'HIGH': '#dc3545', 'MEDIUM': '#ffc107', 'LOW': '#28a745'},
                        title='Population Stability Index',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)


def section_drift_history():
    """Drift trends over time."""
    st.header("📈 Drift History & Trends")
    
    combined_reports = get_combined_monitoring_history(limit=50)
    
    if not combined_reports:
        st.info("No monitoring history available")
        return
    
    # Convert to DataFrame for analysis
    history_data = []
    for report in combined_reports:
        custom = report.get('custom_detector', {})
        alerting = report.get('alerting', {})
        history_data.append({
            'Timestamp': report['timestamp'],
            'Status': custom.get('overall_status', 'UNKNOWN'),
            'High Severity': custom.get('summary', {}).get('high_severity', 0),
            'Medium Severity': custom.get('summary', {}).get('medium_severity', 0),
            'Samples': report.get('run_metadata', {}).get('samples_analyzed', 0),
            'Alert Triggered': bool(alerting.get('alert_triggered', False)),
            'Alert Level': alerting.get('alert_level', 'NONE')
        })
    
    df_history = pd.DataFrame(history_data)
    
    if not df_history.empty:
        df_history['Timestamp'] = pd.to_datetime(df_history['Timestamp'])
        df_history = df_history.sort_values('Timestamp')
        
        # Status trend
        st.subheader("Drift Status Over Time")
        
        status_colors = {
            'CRITICAL': '#dc3545',
            'WARNING': '#ffc107',
            'OK': '#28a745',
            'UNKNOWN': '#999999'
        }
        
        fig = go.Figure()
        for status in ['CRITICAL', 'WARNING', 'OK']:
            mask = df_history['Status'] == status
            fig.add_trace(go.Scatter(
                x=df_history[mask]['Timestamp'],
                y=df_history[mask]['Status'],
                mode='markers',
                name=status,
                marker=dict(size=10, color=status_colors.get(status, '#999999')),
                text=df_history[mask]['Status'],
                hovertemplate='<b>%{text}</b><br>%{x}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Monitoring Status Timeline',
            xaxis_title='Date',
            yaxis_title='Status',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature drift count trend
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("High Severity Features Over Time")
            fig = px.line(
                df_history,
                x='Timestamp',
                y='High Severity',
                markers=True,
                title='High Severity Drift Count',
                height=400
            )
            fig.update_traces(line=dict(color='#dc3545', width=3))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Medium Severity Features Over Time")
            fig = px.line(
                df_history,
                x='Timestamp',
                y='Medium Severity',
                markers=True,
                title='Medium Severity Drift Count',
                height=400
            )
            fig.update_traces(line=dict(color='#ffc107', width=3))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Alert Events Over Time")
        alert_events = df_history[df_history['Alert Triggered']]
        if not alert_events.empty:
            fig = px.scatter(
                alert_events,
                x='Timestamp',
                y='Alert Level',
                color='Alert Level',
                color_discrete_map={'CRITICAL': '#dc3545', 'WARNING': '#ffc107', 'NONE': '#28a745'},
                title='Triggered Alerts Timeline',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No alerts triggered in selected history window.")
        
        # Statistics
        st.subheader("Historical Statistics")
        
        stats_data = {
            'Metric': [
                'Total Monitoring Runs',
                'Critical Alerts',
                'Warning Alerts',
                'OK Statuses',
                'Avg High Severity Features',
                'Avg Medium Severity Features',
                'Triggered Alerts'
            ],
            'Value': [
                str(len(df_history)),
                str(len(df_history[df_history['Status'] == 'CRITICAL'])),
                str(len(df_history[df_history['Status'] == 'WARNING'])),
                str(len(df_history[df_history['Status'] == 'OK'])),
                f"{df_history['High Severity'].mean():.2f}",
                f"{df_history['Medium Severity'].mean():.2f}",
                str(int(df_history['Alert Triggered'].sum()))
            ]
        }
        
        st.dataframe(
            pd.DataFrame(stats_data),
            width='stretch',
            hide_index=True
        )


def section_predictions():
    """Prediction outputs with PNG reports only."""
    st.header("🧾 Prediction Section")
    st.markdown(
        """
        <div class="hero-card">
            <h3 style="margin:0;">PNG Reports Only</h3>
            <p style="margin:6px 0 0 0;">Single and batch prediction reporting is visual-only here: newest PNG per prediction type.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    latest_single_png = get_latest_prediction_png(patterns=['single_prediction_*.png'])
    latest_batch_png = get_latest_prediction_png(patterns=['batch_prediction_*.png'])

    if latest_single_png is None and latest_batch_png is None:
        st.info("No PNG prediction reports found. Run single or batch prediction cells to generate PNG files in data/predictions/.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Newest Single Prediction PNG')
        if latest_single_png is not None:
            single_time = datetime.fromtimestamp(latest_single_png.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            st.markdown(
                f"""
                <div class="artifact-card"><b>File:</b> {latest_single_png.name} &nbsp;|&nbsp; <b>Updated:</b> {single_time}</div>
                """,
                unsafe_allow_html=True,
            )
            st.image(str(latest_single_png), caption='Latest single prediction report', use_container_width=True)
        else:
            st.info('No single prediction PNG found yet.')

    with col2:
        st.subheader('Newest Batch Prediction PNG')
        if latest_batch_png is not None:
            batch_time = datetime.fromtimestamp(latest_batch_png.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            st.markdown(
                f"""
                <div class="artifact-card"><b>File:</b> {latest_batch_png.name} &nbsp;|&nbsp; <b>Updated:</b> {batch_time}</div>
                """,
                unsafe_allow_html=True,
            )
            st.image(str(latest_batch_png), caption='Latest batch prediction report', use_container_width=True)
        else:
            st.info('No batch prediction PNG found yet.')


def section_visual_lab():
    """Professional drift visual lab with top features and side-by-side distributions."""
    st.header("🧪 Drift Visual Lab")

    frames = load_visualization_frames()
    if frames is None:
        st.info('Unable to load numeric datasets for KDE/violin/heatmap visualizations.')
        return

    reference_df = frames['reference_df']
    current_df = frames['current_df']
    numeric_cols = frames['numeric_cols']
    payload = get_latest_drift_payload()

    st.caption(
        f"Reference: {frames['reference_path']} | Current: {frames['current_path']}"
    )

    ranked = rank_drift_features(payload, top_n=4) if payload is not None else {'numeric': [], 'categorical': []}
    candidate_numeric = [x['feature'] for x in ranked['numeric'] if x['feature'] in numeric_cols]
    if not candidate_numeric:
        candidate_numeric = numeric_cols[:4]

    st.subheader("Numeric Drift: Violin Plots")
    violin_cols = st.columns(2)
    for idx, feature in enumerate(candidate_numeric[:4]):
        with violin_cols[idx % 2]:
            ref_vals = pd.to_numeric(reference_df[feature], errors='coerce').dropna().values
            cur_vals = pd.to_numeric(current_df[feature], errors='coerce').dropna().values
            violin_df = pd.DataFrame({
                'value': np.concatenate([ref_vals, cur_vals]) if len(ref_vals) and len(cur_vals) else [],
                'dataset': (['Reference'] * len(ref_vals)) + (['Drifted'] * len(cur_vals)),
            })
            if violin_df.empty:
                st.info(f"No valid values for {feature}")
                continue

            fig_violin = px.violin(
                violin_df,
                x='dataset',
                y='value',
                color='dataset',
                box=True,
                points='outliers',
                color_discrete_map={'Reference': '#00d2ff', 'Drifted': '#ff6b6b'},
                title=f"{feature}: Distribution Shape",
                height=320,
            )
            fig_violin.update_layout(template='plotly_dark', margin=dict(l=20, r=20, t=48, b=20))
            st.plotly_chart(fig_violin, use_container_width=True)

    st.subheader("Numeric Drift: Histogram Overlays")
    hist_cols = st.columns(2)
    for idx, feature in enumerate(candidate_numeric[:4]):
        with hist_cols[idx % 2]:
            ref_vals = pd.to_numeric(reference_df[feature], errors='coerce').dropna().values
            cur_vals = pd.to_numeric(current_df[feature], errors='coerce').dropna().values
            hist_df = pd.DataFrame({
                'value': np.concatenate([ref_vals, cur_vals]) if len(ref_vals) and len(cur_vals) else [],
                'dataset': (['Reference'] * len(ref_vals)) + (['Drifted'] * len(cur_vals)),
            })
            if hist_df.empty:
                st.info(f"No valid values for {feature}")
                continue

            fig_hist = px.histogram(
                hist_df,
                x='value',
                color='dataset',
                barmode='overlay',
                nbins=24,
                opacity=0.72,
                color_discrete_map={'Reference': '#00d2ff', 'Drifted': '#ff6b6b'},
                title=f"{feature}: Reference vs Drifted",
                height=320,
            )
            fig_hist.update_layout(template='plotly_dark', margin=dict(l=20, r=20, t=48, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Categorical Drift: Top Shifted Features")
    ranked_categorical = ranked.get('categorical', [])
    categorical_charted = 0
    cat_cols = st.columns(2)
    for item in ranked_categorical:
        feature = item['feature']
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue

        ref_props = reference_df[feature].astype(str).value_counts(normalize=True)
        cur_props = current_df[feature].astype(str).value_counts(normalize=True)
        categories = sorted(set(ref_props.index).union(set(cur_props.index)))[:8]
        chart_df = pd.DataFrame({
            'Category': categories,
            'Reference': [ref_props.get(c, 0) for c in categories],
            'Drifted': [cur_props.get(c, 0) for c in categories],
        })

        melted = chart_df.melt(id_vars='Category', var_name='Dataset', value_name='Proportion')
        with cat_cols[categorical_charted % 2]:
            fig_cat = px.bar(
                melted,
                x='Category',
                y='Proportion',
                color='Dataset',
                barmode='group',
                color_discrete_map={'Reference': '#00d2ff', 'Drifted': '#ff6b6b'},
                title=f"{feature}: Category Proportion Shift",
                height=340,
            )
            fig_cat.update_layout(template='plotly_dark', margin=dict(l=20, r=20, t=48, b=20))
            st.plotly_chart(fig_cat, use_container_width=True)

        categorical_charted += 1
        if categorical_charted >= 4:
            break

    if categorical_charted == 0:
        st.info("No categorical features available for drift charting.")


def section_feature_snapshots():
    """Show old vs drifted snapshots for top drift features."""
    st.header("🧾 Baseline vs Drifted Feature Snapshots")

    frames = load_visualization_frames()
    payload = get_latest_drift_payload()
    if frames is None or payload is None:
        st.info("Unable to load drift snapshots. Ensure reports and datasets are available.")
        return

    reference_df = frames['reference_df']
    current_df = frames['current_df']
    numeric_cols = set(frames['numeric_cols'])

    ranked = rank_drift_features(payload, top_n=4)
    numeric_top = [x['feature'] for x in ranked['numeric']]
    categorical_top = [x['feature'] for x in ranked['categorical']]

    combined_top = (numeric_top + categorical_top)[:4]
    if not combined_top:
        combined_top = list(reference_df.columns[:4])

    st.subheader("Data Preview: Old vs Drifted")
    preview_cols = [f for f in combined_top if f in reference_df.columns and f in current_df.columns]
    if not preview_cols:
        st.info("No shared top features available for preview.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Old Data (Reference)")
        st.dataframe(reference_df[preview_cols].head(8), use_container_width=True, hide_index=True)
    with col2:
        st.caption("Drifted Data (Current)")
        st.dataframe(current_df[preview_cols].head(8), use_container_width=True, hide_index=True)

    st.subheader("Feature-by-Feature Comparison")
    rows = []
    for feature in preview_cols:
        if feature in numeric_cols:
            ref_series = pd.to_numeric(reference_df[feature], errors='coerce').dropna()
            cur_series = pd.to_numeric(current_df[feature], errors='coerce').dropna()
            rows.append({
                'Feature': feature,
                'Type': 'Numeric',
                'Reference': f"mean={ref_series.mean():.2f}, std={ref_series.std():.2f}",
                'Drifted': f"mean={cur_series.mean():.2f}, std={cur_series.std():.2f}",
            })
        else:
            ref_top = reference_df[feature].astype(str).value_counts(normalize=True).head(1)
            cur_top = current_df[feature].astype(str).value_counts(normalize=True).head(1)
            ref_desc = f"top={ref_top.index[0]} ({ref_top.iloc[0]:.1%})" if not ref_top.empty else 'n/a'
            cur_desc = f"top={cur_top.index[0]} ({cur_top.iloc[0]:.1%})" if not cur_top.empty else 'n/a'
            rows.append({
                'Feature': feature,
                'Type': 'Categorical',
                'Reference': ref_desc,
                'Drifted': cur_desc,
            })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def section_drift_hub():
    """Dedicated drift area with separated drift subsections."""
    st.header("📡 Drift Section")
    section_drift_command_center()

    drift_tab_1, drift_tab_2, drift_tab_3, drift_tab_4, drift_tab_5 = st.tabs(
        ["Overview", "Features", "History", "Visual Lab", "Comparisons"]
    )

    with drift_tab_1:
        section_overview()
    with drift_tab_2:
        section_feature_details()
    with drift_tab_3:
        section_drift_history()
    with drift_tab_4:
        section_visual_lab()
    with drift_tab_5:
        section_feature_snapshots()


def section_settings():
    """Configuration and settings."""
    st.header("⚙️ Settings & Configuration")
    
    st.subheader("Drift Detection Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Numeric Features:**
        - Z-Score Warning: 1.5 std
        - Z-Score Critical: 2.5 std
        - KS Warning: 0.3
        - KS Critical: 0.5
        - PSI Warning: 0.1
        - PSI Critical: 0.25
        """)
    
    with col2:
        st.info("""
        **Categorical Features:**
        - Proportion Warning: 20%
        - Proportion Critical: 35%
        - Chi-Square p-value: 0.01
        - PSI Warning: 0.1
        - PSI Critical: 0.25
        """)
    
    st.subheader("Data Paths")
    st.code("""
History Directory: data/drift_history/
Training Stats: data/processed/training_reference_stats.json
Baseline Data: data/user_testing/baseline_large.csv
Predictions Directory: data/predictions/
    """)
    
    st.subheader("Usage Commands")
    
    st.code("""
# Run monitoring
python src/run_monitoring.py --data new_batch.csv

# Alert only on critical drift
python src/run_monitoring.py --data new_batch.csv --alert-on critical

# Send alerts to webhook
python src/run_monitoring.py --data new_batch.csv --webhook-url https://example/webhook

# Disable alert dispatch
python src/run_monitoring.py --data new_batch.csv --disable-alerts

# Run dashboard
streamlit run src/dashboard.py

# Quick drift check
python src/drift_detector.py
    """, language="bash")


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main dashboard app."""
    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin:0;">Data Drift Command Center</h1>
            <p style="margin:8px 0 0 0;">Color-rich monitoring for drift, predictions, and alert diagnostics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pages = ["🏠 Home", "📡 Drift Section", "🧾 Prediction Section", "⚙️ Settings"]
    if 'window_index' not in st.session_state:
        st.session_state.window_index = 0

    # Top sliding-window controls
    nav_col, prev_col, next_col, refresh_col = st.columns([7.5, 1.2, 1.2, 1.6])

    with nav_col:
        selected = st.radio(
            "Window",
            pages,
            index=st.session_state.window_index,
            horizontal=True,
            label_visibility="collapsed",
            key="window_selector",
        )
        st.session_state.window_index = pages.index(selected)

    with prev_col:
        if st.button("◀", use_container_width=True, help="Previous panel"):
            st.session_state.window_index = (st.session_state.window_index - 1) % len(pages)
            st.rerun()

    with next_col:
        if st.button("▶", use_container_width=True, help="Next panel"):
            st.session_state.window_index = (st.session_state.window_index + 1) % len(pages)
            st.rerun()

    with refresh_col:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    page = pages[st.session_state.window_index]

    # Compact supporting info in expandable side panel.
    with st.sidebar:
        st.title("Panel Info")
        st.caption("Swipe-like controls are available at the top with ◀ and ▶.")
        with st.expander("Monitoring Notes", expanded=False):
            st.markdown("""
            **Drift Detection Techniques:**
            - **Z-Score**: Mean shift detection
            - **KS Test**: Distribution matching
            - **PSI**: Population Stability Index
            - **Chi-Square**: Categorical proportions

            **Status Levels:**
            - 🚨 **CRITICAL**: Significant drift (action required)
            - ⚠️ **WARNING**: Moderate drift (monitor)
            - ✓ **OK**: No significant drift
            """)

    st.markdown(
        f"""
        <div class="window-shell">
            <div class="window-header">
                <div class="window-dots">
                    <span class="window-dot" style="background:#ff5f57;"></span>
                    <span class="window-dot" style="background:#ffbd2e;"></span>
                    <span class="window-dot" style="background:#28c840;"></span>
                </div>
                <span class="window-title">Active Window: {page}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Route pages within animated shell
    if page == "🏠 Home":
        section_home()
    elif page == "📡 Drift Section":
        section_drift_hub()
    elif page == "🧾 Prediction Section":
        section_predictions()
    elif page == "⚙️ Settings":
        section_settings()


if __name__ == "__main__":
    main()
