"""Microsoft Teams alerting for monitoring violations.

Tiered alerting:
    Tier 1 — Log only (all metrics to DB, visible on dashboard).
    Tier 2 — Warning: sustained drift for 3+ of last 5 days → Teams.
    Tier 3 — Critical: any Level 1 violation or extreme PSI → immediate Teams.

Setup:
    1. In Teams, create an Incoming Webhook connector for your channel.
    2. Set the webhook URL in your .env: TEAMS_WEBHOOK_URL=https://...
    3. The Alerter reads this automatically.

Usage:
    alerter = Alerter()
    alerter.send(channel="#fraud-alerts", message="...", severity="critical")
"""

from __future__ import annotations

import json
import os
import urllib.request

from src.logger.logger import get_logger

logger = get_logger("alerter")


class Alerter:
    """Sends alerts to Microsoft Teams via incoming webhook.

    The webhook URL is read from the TEAMS_WEBHOOK_URL environment variable.
    If not set, alerts are logged but not sent (dry mode).
    """

    def __init__(self) -> None:
        self.webhook_url = os.environ.get("TEAMS_WEBHOOK_URL", "")
        if not self.webhook_url:
            logger.info("TEAMS_WEBHOOK_URL not set — alerts will be logged only")

    def send(
        self,
        channel: str,
        message: str,
        severity: str = "warning",
        project_id: str = "",
    ) -> bool:
        """Sends an alert message.

        Args:
            channel: Teams channel name (for logging context; actual routing
                     is determined by the webhook).
            message: The alert message body.
            severity: "critical", "warning", or "info".
            project_id: Optional project ID for context.

        Returns:
            True if the alert was sent (or logged in dry mode), False on error.
        """
        prefix = _severity_prefix(severity)
        full_message = (
            f"{prefix} [{severity.upper()}] "
            f"{f'{project_id} — ' if project_id else ''}"
            f"{message}"
        )

        # Always log
        log_fn = logger.warning if severity == "critical" else logger.info
        log_fn(f"Alert ({channel}): {full_message}")

        # Send to Teams if webhook is configured
        if self.webhook_url:
            return self._send_teams(full_message, severity, project_id)

        return True

    def send_check_results(
        self,
        project_id: str,
        channel: str,
        failed_checks: list,
        run_date: str,
    ) -> bool:
        """Sends a summary of failed monitoring checks.

        Args:
            project_id: The project that failed checks.
            channel: Teams channel name (for logging context).
            failed_checks: List of CheckResult objects that failed.
            run_date: Date of the inference run.

        Returns:
            True if sent successfully.
        """
        if not failed_checks:
            return True

        lines = [f"Monitoring alerts for **{project_id}** on `{run_date}`:"]
        for check in failed_checks:
            lines.append(f"- {check.check_name}: {check.message}")

        return self.send(
            channel=channel,
            message="\n".join(lines),
            severity="critical",
            project_id=project_id,
        )

    def _send_teams(self, message: str, severity: str, project_id: str) -> bool:
        """Posts a message to Microsoft Teams via incoming webhook.

        Uses the Adaptive Card format for rich formatting.

        Args:
            message: Plain text message.
            severity: Alert severity level.
            project_id: Project that triggered the alert.

        Returns:
            True if the webhook call succeeded.
        """
        color = {"critical": "attention", "warning": "warning", "info": "good"}.get(
            severity, "default"
        )

        # Adaptive Card payload for Teams
        payload = json.dumps({
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": f"MLOps Alert — {project_id or 'Platform'}",
                                "weight": "Bolder",
                                "size": "Medium",
                                "color": color,
                            },
                            {
                                "type": "TextBlock",
                                "text": message,
                                "wrap": True,
                            },
                        ],
                    },
                }
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status in (200, 202):
                    logger.info("Teams alert sent successfully")
                    return True
                logger.warning(f"Teams returned status {resp.status}")
                return False
        except Exception as e:
            logger.error(f"Failed to send Teams alert: {e}")
            return False


def _severity_prefix(severity: str) -> str:
    """Returns a text prefix for the alert severity."""
    return {
        "critical": "[!!!]",
        "warning": "[!]",
        "info": "[i]",
    }.get(severity, "[?]")
