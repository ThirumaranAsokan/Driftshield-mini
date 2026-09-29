"""CLI for inspecting DriftShield data — alerts, traces, baselines, and audit export."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from driftshield_mini.embeddings import download_model
from driftshield_mini.export import export_drift_events, export_traces
from driftshield_mini.storage import TraceStore

console = Console()


def _parse_time_window(window: str) -> float:
    """Parse '24h', '7d', '30m' etc into a timestamp."""
    unit = window[-1].lower()
    value = float(window[:-1])
    multipliers = {"m": 60, "h": 3600, "d": 86400}
    if unit not in multipliers:
        raise click.BadParameter(f"Unknown time unit '{unit}'. Use m/h/d.")
    return time.time() - (value * multipliers[unit])


def _format_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


SEVERITY_STYLES = {
    "LOW": "green",
    "MED": "yellow",
    "HIGH": "dark_orange",
    "CRITICAL": "bold red",
}


@click.group()
@click.option("--db", default=None, help="Path to DriftShield database")
@click.pass_context
def cli(ctx: click.Context, db: str | None) -> None:
    """DriftShield — Real-time drift detection for AI agents."""
    ctx.ensure_object(dict)
    ctx.obj["store"] = TraceStore(db_path=db)


@cli.command()
@click.option("--last", default="24h", help="Time window (e.g. 24h, 7d, 30m)")
@click.option("--agent", default=None, help="Filter by agent ID")
@click.option("--severity", default=None, type=click.Choice(["LOW", "MED", "HIGH", "CRITICAL"]))
@click.option("--limit", default=20, help="Max results")
@click.pass_context
def alerts(ctx: click.Context, last: str, agent: str | None,
           severity: str | None, limit: int) -> None:
    """View recent drift alerts."""
    store: TraceStore = ctx.obj["store"]
    since = _parse_time_window(last)
    events = store.get_drift_events(agent_id=agent, since=since, severity=severity, limit=limit)

    if not events:
        console.print("[dim]No drift events found.[/dim]")
        return

    for event in events:
        style = SEVERITY_STYLES.get(event.severity, "white")
        console.print(f"\n  [{style}][{event.severity}][/{style}]  {_format_ts(event.timestamp)}  [bold]{event.agent_id}[/bold]")
        console.print(f"          {event.message}")
        console.print(f"          [dim]Suggested action: {event.suggested_action}[/dim]")

    console.print(f"\n[dim]{len(events)} alert(s) shown[/dim]")


@cli.command()
@click.argument("agent_id")
@click.option("--run", default="latest", help="Run ID or 'latest'")
@click.option("--limit", default=50, help="Max events")
@click.pass_context
def traces(ctx: click.Context, agent_id: str, run: str, limit: int) -> None:
    """View trace events for an agent run."""
    store: TraceStore = ctx.obj["store"]

    if run == "latest":
        run_ids = store.get_run_ids(agent_id, limit=1)
        if not run_ids:
            console.print(f"[dim]No runs found for agent '{agent_id}'[/dim]")
            return
        run = run_ids[0]

    events = store.get_run_traces(agent_id, run)
    if not events:
        console.print(f"[dim]No traces found for run '{run}'[/dim]")
        return

    console.print(f"\n[bold]Agent:[/bold] {agent_id}  [bold]Run:[/bold] {run}\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Time", width=20)
    table.add_column("Type", width=16)
    table.add_column("Action", width=24)
    table.add_column("Tokens", width=8, justify="right")
    table.add_column("Duration", width=10, justify="right")

    for e in events[:limit]:
        table.add_row(
            _format_ts(e.timestamp),
            e.action_type,
            e.action_name,
            str(e.token_count) if e.token_count else "-",
            f"{e.duration_ms:.0f}ms" if e.duration_ms else "-",
        )

    console.print(table)
    console.print(f"\n[dim]{len(events)} event(s) in run[/dim]")


@cli.command()
@click.argument("agent_id")
@click.pass_context
def baseline(ctx: click.Context, agent_id: str) -> None:
    """Show baseline statistics for an agent."""
    store: TraceStore = ctx.obj["store"]
    bl = store.get_baseline(agent_id)

    if not bl:
        console.print(f"[dim]No baseline found for agent '{agent_id}'[/dim]")
        return

    status = "[green]CALIBRATED[/green]" if bl.is_calibrated else "[yellow]PENDING[/yellow]"

    console.print(f"\n[bold]Baseline for '{agent_id}'[/bold]  {status}")
    console.print(f"  Calibration runs: {bl.calibration_runs}")
    console.print(f"  Tokens/run:       {bl.mean_tokens_per_run:.0f} ± {bl.std_tokens_per_run:.0f}")
    console.print(f"  Tools/run:        {bl.mean_tools_per_run:.1f} ± {bl.std_tools_per_run:.1f}")
    console.print(f"  Duration/run:     {bl.mean_duration_ms:.0f}ms ± {bl.std_duration_ms:.0f}ms")

    if bl.common_sequences:
        console.print(f"\n  Common sequences:")
        for seq in bl.common_sequences[:5]:
            console.print(f"    {'  →  '.join(seq)}")

    console.print()


@cli.command()
@click.argument("agent_id")
@click.option("--limit", default=10, help="Number of runs to show")
@click.pass_context
def runs(ctx: click.Context, agent_id: str, limit: int) -> None:
    """List recent runs for an agent."""
    store: TraceStore = ctx.obj["store"]
    run_ids = store.get_run_ids(agent_id, limit=limit)

    if not run_ids:
        console.print(f"[dim]No runs found for agent '{agent_id}'[/dim]")
        return

    console.print(f"\n[bold]Recent runs for '{agent_id}'[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Run ID", width=16)
    table.add_column("Events", width=8, justify="right")
    table.add_column("Tokens", width=10, justify="right")
    table.add_column("Tools", width=8, justify="right")
    table.add_column("Duration", width=12, justify="right")

    for run_id in run_ids:
        stats = store.get_run_stats(agent_id, run_id)
        duration = stats["total_duration_ms"]
        table.add_row(
            run_id,
            str(stats["event_count"]),
            f"{stats['total_tokens']:,}",
            str(stats["tool_calls"]),
            f"{duration:.0f}ms" if duration else "-",
        )

    console.print(table)
    console.print()


@cli.command()
@click.option("--agent", default=None, help="Filter by agent ID")
@click.option("--output", required=True, help="Output file path (e.g. audit.csv or audit.json)")
@click.option("--format", "fmt", type=click.Choice(["csv", "json"]), default=None,
              help="Export format (default: inferred from file extension)")
@click.option("--last", default=None, help="Only export last N h/d (e.g. 24h, 7d)")
@click.option("--drift-only", is_flag=True, help="Export drift events only (incident log)")
@click.option("--limit", default=100000, help="Max rows")
@click.pass_context
def export(ctx: click.Context, agent: str | None, output: str, fmt: str | None,
           last: str | None, drift_only: bool, limit: int) -> None:
    """Export an audit log for compliance (FCA / EU AI Act)."""
    store: TraceStore = ctx.obj["store"]

    if fmt is None:
        fmt = Path(output).suffix.lstrip(".").lower() or "csv"
    hours = (time.time() - _parse_time_window(last)) / 3600 if last else None

    if drift_only:
        path = export_drift_events(store, output, agent_id=agent, fmt=fmt, hours=hours, limit=limit)
    else:
        path = export_traces(store, output, agent_id=agent, fmt=fmt, hours=hours, limit=limit)

    console.print(f"[green]OK[/green] Exported to {path}")


@cli.command(name="download-model")
@click.option("--output", default=None, help="Target directory (default: bundled package dir)")
def download_model_cmd(output: str | None) -> None:
    """Download the embedding model for offline/air-gapped use."""
    path = download_model(Path(output) if output else None)
    console.print(f"[green]OK[/green] Model saved to {path}")


if __name__ == "__main__":
    cli()
