"""Regression tests for alert severity handling."""

from driftshield_mini.alerts import AlertDispatcher


def test_alert_dispatcher_defaults_to_medium_severity():
    dispatcher = AlertDispatcher()
    try:
        assert dispatcher.min_severity == "MED"
    finally:
        dispatcher.close()
