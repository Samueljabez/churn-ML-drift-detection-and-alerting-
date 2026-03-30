"""
Drift Monitoring Orchestration Script

Runs custom drift detection and triggers alerts based on drift severity.
Can be run manually or from an external scheduler.

Usage:
    python src/run_monitoring.py --data new_batch.csv
    python src/run_monitoring.py --data new_batch.csv --alert-on critical
    python src/run_monitoring.py --data new_batch.csv --webhook-url https://example/webhook
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from alerting import AlertDispatcher
from drift_detector import DriftDetector


class DriftMonitor:
    """Orchestrates drift detection and alerting."""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        training_stats_path: Optional[str] = None,
        target_column: Optional[str] = None,
        history_dir: Optional[str] = None,
        alerts_enabled: bool = True,
        alert_on: str = "warning",
        webhook_url: Optional[str] = None,
    ):
        self.reference_data = reference_data
        self.target_column = target_column
        self.history_dir = Path(history_dir) if history_dir else (Path(__file__).parent.parent / "data" / "drift_history")
        self.history_dir.mkdir(parents=True, exist_ok=True)

        if training_stats_path:
            self.custom_detector = DriftDetector(stats_path=training_stats_path)
        else:
            self.custom_detector = DriftDetector()
        self.alert_on = alert_on
        self.alert_dispatcher = AlertDispatcher(enabled=alerts_enabled, webhook_url=webhook_url)

    def run_monitoring(
        self,
        current_data: pd.DataFrame,
        batch_name: Optional[str] = None,
        input_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run custom drift monitoring and trigger alert if configured threshold is reached."""
        timestamp = datetime.now()
        batch_id = batch_name or timestamp.strftime("%Y%m%d_%H%M%S")

        print("\n" + "=" * 100)
        print(f"DRIFT MONITORING RUN - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Batch ID: {batch_id}")
        print("=" * 100)

        print("\n[1/2] Running custom drift detector...")
        custom_report = self.custom_detector.detect_drift(current_data)
        self.custom_detector.print_report(custom_report)
        custom_summary = self.custom_detector.get_report_summary(custom_report)

        print("\n[2/2] Exporting reports and evaluating alert conditions...")
        custom_report_path = self.custom_detector.export_report(custom_report)
        print(f"✓ Custom detector report saved to: {custom_report_path}")

        alert_info = self._evaluate_alert(custom_report, custom_summary, batch_id, timestamp)
        alert_delivery: Dict[str, Any] = {
            "enabled": self.alert_dispatcher.enabled,
            "console_sent": False,
            "webhook_sent": False,
            "webhook_error": None,
        }

        if alert_info["alert_triggered"]:
            alert_payload: Dict[str, Any] = {
                "batch_id": batch_id,
                "timestamp": timestamp.isoformat(),
                "level": alert_info["alert_level"],
                "status": custom_report.get("overall_status", "UNKNOWN"),
                "message": alert_info["alert_message"],
                "summary": custom_summary,
            }
            alert_delivery = self.alert_dispatcher.send_alert(
                level=alert_info["alert_level"],
                message=alert_info["alert_message"],
                payload=alert_payload,
            )

        combined_report: Dict[str, Any] = {
            "batch_id": batch_id,
            "timestamp": timestamp.isoformat(),
            "run_metadata": {
                "monitor": "DriftMonitor",
                "run_mode": "manual_cli",
                "samples_analyzed": len(current_data),
                "reference_samples": len(self.reference_data),
                "alert_policy": self.alert_on,
            },
            "input_metadata": input_metadata or {},
            "custom_detector": {
                "overall_status": custom_report["overall_status"],
                "summary": custom_summary,
                "detailed_report": custom_report_path,
            },
            "alerting": {
                **alert_info,
                **alert_delivery,
            },
            "artifacts": {
                "custom_report_json": custom_report_path,
            },
            "combined_verdict": self._combine_verdicts(custom_report["overall_status"], alert_info),
        }

        combined_report_path = self._save_combined_report(combined_report, batch_id=batch_id)

        print("\n" + "=" * 100)
        print("MONITORING COMPLETE")
        print(f"Combined verdict: {combined_report['combined_verdict']}")
        print(f"Alert triggered: {combined_report['alerting']['alert_triggered']}")
        print(f"Full report: {combined_report_path}")
        print("=" * 100 + "\n")

        return combined_report

    def _evaluate_alert(
        self,
        custom_report: Dict[str, Any],
        custom_summary: Dict[str, Any],
        batch_id: str,
        timestamp: datetime,
    ) -> Dict[str, Any]:
        """Compute alert decision based on overall drift status and configured policy."""
        status = custom_report.get("overall_status", "UNKNOWN")
        high = custom_summary.get("high_severity", 0)
        medium = custom_summary.get("medium_severity", 0)

        alert_triggered = False
        alert_level = "NONE"

        if status == "CRITICAL":
            alert_triggered = True
            alert_level = "CRITICAL"
        elif status == "WARNING" and self.alert_on == "warning":
            alert_triggered = True
            alert_level = "WARNING"

        alert_message = (
            f"Drift status={status} | high={high}, medium={medium} | batch={batch_id}"
            if alert_triggered
            else "No alert triggered for current drift status"
        )

        return {
            "alert_triggered": alert_triggered,
            "alert_level": alert_level,
            "alert_message": alert_message,
            "status": status,
            "high_severity": high,
            "medium_severity": medium,
            "evaluated_at": timestamp.isoformat(),
        }

    @staticmethod
    def _combine_verdicts(custom_status: str, alert_info: Dict[str, Any]) -> str:
        if alert_info.get("alert_triggered"):
            if alert_info.get("alert_level") == "CRITICAL":
                return "CRITICAL - Drift alert triggered. Immediate action recommended."
            return "WARNING - Drift alert triggered. Monitor and investigate."

        if custom_status == "CRITICAL":
            return "CRITICAL - Significant drift detected. Alert policy suppressed notification."
        if custom_status == "WARNING":
            return "WARNING - Moderate drift detected. Alert policy suppressed notification."
        return "OK - No significant drift detected."

    def _save_combined_report(self, combined_report: Dict[str, Any], batch_id: str) -> str:
        filepath = self.history_dir / f"combined_monitoring_report_{batch_id}.json"
        with open(filepath, "w") as f:
            json.dump(combined_report, f, indent=2, default=str)
        return str(filepath)

    @staticmethod
    def load_monitoring_history(history_dir: Optional[Union[str, Path]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        history_path = Path(history_dir) if history_dir is not None else (Path(__file__).parent.parent / "data" / "drift_history")

        if not history_path.exists():
            return []

        reports: List[Dict[str, Any]] = []
        for filepath in sorted(history_path.glob("combined_monitoring_report_*.json"), reverse=True)[:limit]:
            with open(filepath, "r") as f:
                reports.append(json.load(f))

        return reports


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run drift monitoring with custom detector and alerting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/run_monitoring.py --data new_batch.csv
  python src/run_monitoring.py --data data.csv --alert-on critical
  python src/run_monitoring.py --data data.csv --webhook-url https://example/webhook
        """,
    )

    parser.add_argument("--data", required=True, type=str, help="Path to new data batch to monitor")
    parser.add_argument("--reference", type=str, default=None, help="Path to reference/baseline data (default: auto-loads)")
    parser.add_argument("--stats", type=str, default=None, help="Path to training statistics JSON")
    parser.add_argument("--target", type=str, default=None, help="Target column name")
    parser.add_argument("--batch-name", type=str, default=None, help="Batch identifier")
    parser.add_argument("--history-dir", type=str, default=None, help="Output directory for JSON history")
    parser.add_argument("--alert-on", type=str, choices=["warning", "critical"], default="warning", help="Alert threshold")
    parser.add_argument("--disable-alerts", action="store_true", help="Disable alert dispatch")
    parser.add_argument("--webhook-url", type=str, default=None, help="Webhook URL for alert notifications")

    args = parser.parse_args()

    print("Loading data files...")
    current_data = pd.read_csv(args.data)
    print(f"✓ Loaded current data: {current_data.shape}")

    if args.reference:
        reference_data = pd.read_csv(args.reference)
        reference_path = Path(args.reference)
        print(f"✓ Loaded reference data: {reference_data.shape}")
    else:
        reference_path = Path(__file__).parent.parent / "data" / "user_testing" / "baseline_large.csv"
        reference_data = pd.read_csv(reference_path)
        print(f"✓ Loaded baseline data: {reference_data.shape}")

    print("\nInitializing drift monitor...")
    monitor = DriftMonitor(
        reference_data,
        training_stats_path=args.stats,
        target_column=args.target,
        history_dir=args.history_dir,
        alerts_enabled=not args.disable_alerts,
        alert_on=args.alert_on,
        webhook_url=args.webhook_url,
    )

    report = monitor.run_monitoring(
        current_data,
        batch_name=args.batch_name,
        input_metadata={
            "current_data_path": str(Path(args.data).resolve()),
            "reference_data_path": str(reference_path.resolve()),
            "training_stats_path": str(Path(args.stats).resolve()) if args.stats else None,
            "target_column": args.target,
        },
    )

    print("\n" + "=" * 100)
    print("MONITORING SUMMARY")
    print("=" * 100)
    print(f"Batch: {report['batch_id']}")
    print(f"Timestamp: {report['timestamp']}")
    print(f"Samples: {report['run_metadata']['samples_analyzed']}")
    print(f"\nCustom Detector: {report['custom_detector']['overall_status']}")
    print(f"  High severity features: {report['custom_detector']['summary'].get('high_severity', 0)}")
    print(f"  Medium severity features: {report['custom_detector']['summary'].get('medium_severity', 0)}")
    print(f"\nAlert Triggered: {report['alerting']['alert_triggered']}")
    print(f"  Alert level: {report['alerting']['alert_level']}")
    print(f"  Message: {report['alerting']['alert_message']}")
    if report['alerting'].get('webhook_error'):
        print(f"  Webhook error: {report['alerting']['webhook_error']}")
    print(f"\nVerdict: {report['combined_verdict']}")
    print("=" * 100 + "\n")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
