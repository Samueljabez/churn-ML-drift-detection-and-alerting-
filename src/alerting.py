"""
Alerting utilities for drift monitoring.

Supports:
- Console alerts (always available)
- Webhook alerts (optional)
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import request, error


class AlertDispatcher:
    """Dispatch drift alerts to configured channels."""

    def __init__(
        self,
        enabled: bool = True,
        webhook_url: Optional[str] = None,
        timeout_seconds: int = 10,
        alerts_dir: Optional[str] = None,
    ):
        self.enabled = enabled
        self.webhook_url = webhook_url
        self.timeout_seconds = timeout_seconds
        self.alerts_dir = Path(alerts_dir) if alerts_dir else (Path(__file__).parent.parent / "data" / "alerts")
        self.alerts_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _build_alert_id(level: str, payload: Dict[str, Any], emitted_at: str) -> str:
        batch_id = str(payload.get("batch_id", "unknown"))
        signature_raw = f"{level}|{batch_id}|{emitted_at}"
        digest = hashlib.sha1(signature_raw.encode("utf-8")).hexdigest()[:10]
        return f"ALERT-{level[:1]}-{digest}"

    def _append_alert_log(self, event: Dict[str, Any]) -> str:
        log_path = self.alerts_dir / "alert_events.jsonl"
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(event, default=str) + "\n")
        return str(log_path)

    def send_alert(self, level: str, message: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send alert to available channels and return delivery status."""
        emitted_at = datetime.now(timezone.utc).isoformat()
        alert_id = self._build_alert_id(level=level, payload=payload, emitted_at=emitted_at)
        result: Dict[str, Any] = {
            "enabled": self.enabled,
            "level": level,
            "message": message,
            "alert_id": alert_id,
            "emitted_at": emitted_at,
            "console_sent": False,
            "webhook_sent": False,
            "webhook_error": None,
            "alerts_log": None,
        }

        if not self.enabled:
            return result

        guidance = {
            "CRITICAL": "Immediate triage: investigate data pipeline and evaluate model impact.",
            "WARNING": "Investigate soon: review shifted features and monitor next batches.",
        }.get(level, "Review monitoring details.")

        print("\n" + "#" * 100)
        print(f"DRIFT ALERT EVENT  ::  {alert_id}")
        print(f"LEVEL              ::  {level}")
        print(f"TIME (UTC)         ::  {emitted_at}")
        print(f"MESSAGE            ::  {message}")
        print(f"GUIDANCE           ::  {guidance}")
        print("#" * 100)
        result["console_sent"] = True

        event_payload: Dict[str, Any] = {
            **payload,
            "alert_id": alert_id,
            "emitted_at": emitted_at,
            "guidance": guidance,
        }

        alerts_log_path = self._append_alert_log(
            {
                "alert_id": alert_id,
                "level": level,
                "message": message,
                "emitted_at": emitted_at,
                "payload": event_payload,
            }
        )
        result["alerts_log"] = alerts_log_path

        if self.webhook_url:
            try:
                body = json.dumps(event_payload).encode("utf-8")
                req = request.Request(
                    self.webhook_url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    if 200 <= resp.status < 300:
                        result["webhook_sent"] = True
                    else:
                        result["webhook_error"] = f"Webhook returned status {resp.status}"
            except error.URLError as exc:
                result["webhook_error"] = str(exc)
            except Exception as exc:
                result["webhook_error"] = str(exc)

        return result
