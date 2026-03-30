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
                
                st.dataframe(styled_df, width='stretch', hide_index=True)
                
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
                    st.plotly_chart(fig, width='stretch')
                
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
                    st.plotly_chart(fig, width='stretch')
    
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
                
                st.dataframe(df_categorical, width='stretch', hide_index=True)
                
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
                    st.plotly_chart(fig, width='stretch')
                
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
                    st.plotly_chart(fig, width='stretch')


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
        st.plotly_chart(fig, width='stretch')
        
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
            st.plotly_chart(fig, width='stretch')
        
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
            st.plotly_chart(fig, width='stretch')

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
            st.plotly_chart(fig, width='stretch')
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
            st.image(str(latest_single_png), caption='Latest single prediction report', width='stretch')
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
            st.image(str(latest_batch_png), caption='Latest batch prediction report', width='stretch')
        else:
            st.info('No batch prediction PNG found yet.')


def section_visual_lab():
    """Dedicated advanced drift visualizations: KDE, violin, and heatmap."""
    st.header("🧪 Drift Visual Lab")

    frames = load_visualization_frames()
    if frames is None:
        st.info('Unable to load numeric datasets for KDE/violin/heatmap visualizations.')
        return

    reference_df = frames['reference_df']
    current_df = frames['current_df']
    numeric_cols = frames['numeric_cols']

    st.caption(
        f"Reference: {frames['reference_path']} | Current: {frames['current_path']}"
    )

    # Prefer numerics flagged in latest report; fallback to first numeric feature.
    latest_report = load_latest_report()
    candidate_features: List[str] = []
    if latest_report is not None:
        numeric_info = latest_report.get('numeric_features', {})
        for feature_name, info in numeric_info.items():
            if feature_name in numeric_cols and info.get('severity') in ('HIGH', 'MEDIUM'):
                candidate_features.append(feature_name)

    if not candidate_features:
        candidate_features = numeric_cols[:4]

    feature_for_density = candidate_features[0]

    left, right = st.columns(2)

    with left:
        st.subheader(f"KDE Plot: {feature_for_density}")
        ref_vals = pd.to_numeric(reference_df[feature_for_density], errors='coerce').dropna().values
        cur_vals = pd.to_numeric(current_df[feature_for_density], errors='coerce').dropna().values

        if len(ref_vals) > 1 and len(cur_vals) > 1:
            x_min = min(float(np.min(ref_vals)), float(np.min(cur_vals)))
            x_max = max(float(np.max(ref_vals)), float(np.max(cur_vals)))
            x_grid = np.linspace(x_min, x_max, 250)

            ref_kde = gaussian_kde(ref_vals)(x_grid)
            cur_kde = gaussian_kde(cur_vals)(x_grid)

            fig_kde = go.Figure()
            fig_kde.add_trace(go.Scatter(x=x_grid, y=ref_kde, mode='lines', name='Reference', line=dict(color='#00d2ff', width=3)))
            fig_kde.add_trace(go.Scatter(x=x_grid, y=cur_kde, mode='lines', name='Current', line=dict(color='#ff6b6b', width=3)))
            fig_kde.update_layout(
                title='Kernel Density Comparison',
                xaxis_title=feature_for_density,
                yaxis_title='Density',
                template='plotly_dark',
                height=360,
            )
            st.plotly_chart(fig_kde, width='stretch')
        else:
            st.info('Not enough samples for KDE plot.')

    with right:
        st.subheader(f"Violin Plot: {feature_for_density}")
        violin_df = pd.DataFrame({
            'value': np.concatenate([ref_vals, cur_vals]) if len(ref_vals) > 0 and len(cur_vals) > 0 else [],
            'dataset': (['Reference'] * len(ref_vals)) + (['Current'] * len(cur_vals)),
        })
        if not violin_df.empty:
            fig_violin = px.violin(
                violin_df,
                x='dataset',
                y='value',
                color='dataset',
                box=True,
                points='all',
                color_discrete_map={'Reference': '#00d2ff', 'Current': '#ff6b6b'},
                title='Distribution Shape (Single Feature)',
                height=360,
            )
            fig_violin.update_layout(template='plotly_dark')
            st.plotly_chart(fig_violin, width='stretch')
        else:
            st.info('No valid values for violin plot.')

    st.subheader('Correlation Heatmap')
    heatmap_features = candidate_features[:6] if len(candidate_features) >= 2 else numeric_cols[:6]
    current_numeric = current_df[heatmap_features].apply(pd.to_numeric, errors='coerce')
    corr_df = current_numeric.corr()

    fig_heatmap = px.imshow(
        corr_df,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='Current Batch Feature Correlation Heatmap',
        aspect='auto',
    )
    fig_heatmap.update_layout(template='plotly_dark', height=450)
    st.plotly_chart(fig_heatmap, width='stretch')


def section_drift_hub():
    """Dedicated drift area with separated drift subsections."""
    st.header("📡 Drift Section")
    drift_tab_1, drift_tab_2, drift_tab_3, drift_tab_4 = st.tabs(["Overview", "Features", "History", "Visual Lab"])

    with drift_tab_1:
        section_overview()
    with drift_tab_2:
        section_feature_details()
    with drift_tab_3:
        section_drift_history()
    with drift_tab_4:
        section_visual_lab()


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

    pages = ["📡 Drift Section", "🧾 Prediction Section", "⚙️ Settings"]
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
    if page == "📡 Drift Section":
        section_drift_hub()
    elif page == "🧾 Prediction Section":
        section_predictions()
    elif page == "⚙️ Settings":
        section_settings()


if __name__ == "__main__":
    main()
