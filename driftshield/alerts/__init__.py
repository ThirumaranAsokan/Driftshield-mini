"""Alert dispatchers — Slack, Discord, and generic webhook support."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from driftshield.models import DriftEvent

logger = logging.getLogger(__name__)

# Severity → color mapping
SEVERITY_COLORS = {
    "LOW": "#36a64f",       # green
    "MED": "#daa520",       # amber
    "HIGH": "#ff6600",      # orange
    "CRITICAL": "#ff0000",  # red
}

SEVERITY_EMOJI = {
    "LOW": "🟢",
    "MED": "🟡",
    "HIGH": "🟠",
    "CRITICAL": "🔴",
}


class AlertDispatcher:
    """Dispatches drift alerts via webhook (Slack, Discord, or generic)."""

    def __init__(
        self,
        webhook_url: str | None = None,
        min_severity: str = "MEDIUM",
        cooldown_seconds: float = 60.0,
    ):
        self.webhook_url = webhook_url
        self.min_severity = min_severity
        self.cooldown_seconds = cooldown_seconds
        self._last_alert: dict[str, float] = {}  # agent_id+detector → timestamp

    def should_alert(self, event: DriftEvent) -> bool:
        """Check if alert should fire (severity + cooldown)."""
        if not self.webhook_url:
            return False

        severity_order = ["LOW", "MED", "HIGH", "CRITICAL"]
        min_idx = severity_order.index(self.min_severity) if self.min_severity in severity_order else 1
        event_idx = severity_order.index(event.severity.value) if event.severity.value in severity_order else 0

        if event_idx < min_idx:
            return False

        # Cooldown: don't spam the same detector for the same agent
        key = f"{event.agent_id}:{event.detector.value}"
        now = time.time()
        if key in self._last_alert and (now - self._last_alert[key]) < self.cooldown_seconds:
            return False

        self._last_alert[key] = now
        return True

    async def send_async(self, event: DriftEvent) -> bool:
        """Send alert via async HTTP (preferred)."""
        if not self.should_alert(event):
            return False

        try:
            import httpx

            payload = self._build_payload(event)
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self.webhook_url, json=payload)
                resp.raise_for_status()
                logger.info(f"Alert sent for {event.agent_id}: {event.message}")
                return True
        except Exception as e:
            logger.warning(f"Failed to send alert: {e}")
            return False

    def send_sync(self, event: DriftEvent) -> bool:
        """Send alert via sync HTTP (fallback)."""
        if not self.should_alert(event):
            return False

        try:
            import httpx

            payload = self._build_payload(event)
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(self.webhook_url, json=payload)
                resp.raise_for_status()
                logger.info(f"Alert sent for {event.agent_id}: {event.message}")
                return True
        except Exception as e:
            logger.warning(f"Failed to send alert: {e}")
            return False

    def _build_payload(self, event: DriftEvent) -> dict[str, Any]:
        """Build webhook payload — auto-detects Slack vs Discord vs generic."""
        if not self.webhook_url:
            return {}

        if "hooks.slack.com" in self.webhook_url:
            return self._slack_payload(event)
        elif "discord.com" in self.webhook_url:
            return self._discord_payload(event)
        else:
            return self._generic_payload(event)

    def _slack_payload(self, event: DriftEvent) -> dict[str, Any]:
        emoji = SEVERITY_EMOJI.get(event.severity.value, "⚠️")
        color = SEVERITY_COLORS.get(event.severity.value, "#daa520")
        ts = datetime.fromtimestamp(event.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        return {
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": (
                                    f"{emoji} *[{event.severity.value}] DriftShield Alert*\n"
                                    f"*Agent:* `{event.agent_id}`\n"
                                    f"*Detector:* {event.detector.value}\n"
                                    f"*Time:* {ts}\n\n"
                                    f"{event.message}\n\n"
                                    f"💡 *Suggested action:* {event.suggested_action}"
                                ),
                            },
                        }
                    ],
                }
            ]
        }

    def _discord_payload(self, event: DriftEvent) -> dict[str, Any]:
        emoji = SEVERITY_EMOJI.get(event.severity.value, "⚠️")
        color_hex = SEVERITY_COLORS.get(event.severity.value, "#daa520")
        color_int = int(color_hex.lstrip("#"), 16)
        ts = datetime.fromtimestamp(event.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        return {
            "embeds": [
                {
                    "title": f"{emoji} [{event.severity.value}] DriftShield Alert",
                    "color": color_int,
                    "fields": [
                        {"name": "Agent", "value": f"`{event.agent_id}`", "inline": True},
                        {"name": "Detector", "value": event.detector.value, "inline": True},
                        {"name": "Time", "value": ts, "inline": True},
                        {"name": "Details", "value": event.message, "inline": False},
                        {"name": "💡 Suggested Action", "value": event.suggested_action, "inline": False},
                    ],
                }
            ]
        }

    def _generic_payload(self, event: DriftEvent) -> dict[str, Any]:
        return {
            "source": "driftshield",
            "event": event.to_dict(),
        }
