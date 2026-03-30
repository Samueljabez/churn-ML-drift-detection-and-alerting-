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
from typing import Any, Dict, Optional, cast
from urllib import request, error


class AlertDispatcher:
    """Dispatch drift alerts to configured channels."""

    def __init__(
        self,
        enabled: bool = True,
        webhook_url: Optional[str] = None,
        timeout_seconds: int = 10,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        cooldown_minutes: int = 0,
        webhook_headers: Optional[Dict[str, str]] = None,
        alerts_dir: Optional[str] = None,
    ):
        self.enabled = enabled
        self.webhook_url = webhook_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.cooldown_minutes = max(0, int(cooldown_minutes))
        self.webhook_headers = webhook_headers or {}
        self.alerts_dir = Path(alerts_dir) if alerts_dir else (Path(__file__).parent.parent / "data" / "alerts")
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.alerts_dir / "alert_state.json"

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_file.exists():
            return {}
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return cast(Dict[str, Any], data)
            return {}
        except Exception:
            return {}

    def _save_state(self, state: Dict[str, Any]) -> None:
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    @staticmethod
    def _build_fingerprint(level: str, payload: Dict[str, Any]) -> str:
        status = str(payload.get("status", "UNKNOWN"))
        summary_raw = payload.get("summary")
        if isinstance(summary_raw, dict):
            summary = cast(Dict[str, Any], summary_raw)
        else:
            summary = {}
        high = int(summary.get("high_severity", 0) or 0)
        medium = int(summary.get("medium_severity", 0) or 0)
        base = f"{level}|{status}|{high}|{medium}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def _is_suppressed(self, fingerprint: str, emitted_at: datetime) -> bool:
        if self.cooldown_minutes <= 0:
            return False

        state = self._load_state()
        previous = state.get(fingerprint)
        if not previous:
            state[fingerprint] = emitted_at.isoformat()
            self._save_state(state)
            return False

        try:
            last_sent = datetime.fromisoformat(str(previous))
        except Exception:
            state[fingerprint] = emitted_at.isoformat()
            self._save_state(state)
            return False

        delta_seconds = (emitted_at - last_sent).total_seconds()
        if delta_seconds < self.cooldown_minutes * 60:
            return True

        state[fingerprint] = emitted_at.isoformat()
        self._save_state(state)
        return False

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
        emitted_dt = datetime.now(timezone.utc)
        emitted_at = emitted_dt.isoformat()
        alert_id = self._build_alert_id(level=level, payload=payload, emitted_at=emitted_at)
        fingerprint = self._build_fingerprint(level=level, payload=payload)
        result: Dict[str, Any] = {
            "enabled": self.enabled,
            "level": level,
            "message": message,
            "alert_id": alert_id,
            "emitted_at": emitted_at,
            "suppressed": False,
            "suppression_reason": None,
            "console_sent": False,
            "webhook_sent": False,
            "webhook_error": None,
            "webhook_attempts": 0,
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

        if self._is_suppressed(fingerprint=fingerprint, emitted_at=emitted_dt):
            result["suppressed"] = True
            result["suppression_reason"] = f"cooldown_active_{self.cooldown_minutes}m"

            alerts_log_path = self._append_alert_log(
                {
                    "alert_id": alert_id,
                    "level": level,
                    "message": message,
                    "emitted_at": emitted_at,
                    "suppressed": True,
                    "suppression_reason": result["suppression_reason"],
                    "payload": event_payload,
                }
            )
            result["alerts_log"] = alerts_log_path
            return result

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
            body = json.dumps(event_payload).encode("utf-8")
            headers = {"Content-Type": "application/json", **self.webhook_headers}

            for attempt in range(1, self.max_retries + 2):
                result["webhook_attempts"] = attempt
                try:
                    req = request.Request(
                        self.webhook_url,
                        data=body,
                        headers=headers,
                        method="POST",
                    )
                    with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                        if 200 <= resp.status < 300:
                            result["webhook_sent"] = True
                            result["webhook_error"] = None
                            break
                        result["webhook_error"] = f"Webhook returned status {resp.status}"
                except error.HTTPError as exc:
                    result["webhook_error"] = f"HTTPError {exc.code}: {exc.reason}"
                except error.URLError as exc:
                    result["webhook_error"] = str(exc)
                except Exception as exc:
                    result["webhook_error"] = str(exc)

                if attempt <= self.max_retries and self.retry_backoff_seconds > 0:
                    import time

                    time.sleep(self.retry_backoff_seconds * attempt)

        return result
