# wizard_ops/cli.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from wizard_ops import get_version
from wizard_ops.data_cli import app as data_app
from wizard_ops.hydra_app import main as hydra_main

app = typer.Typer(add_completion=False, help="wizard_ops command-line interface")
app.add_typer(data_app, name="data", help="Data utilities")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(get_version())
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "-v",
        "--version",
        help="Show package version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Top-level CLI entrypoint."""


def _run_hydra(forwarded: list[str]) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], *forwarded]
        hydra_main()
    finally:
        sys.argv = old_argv


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train(ctx: typer.Context) -> None:
    """Train (Hydra-powered)."""
    _run_hydra(["mode=train", *ctx.args])


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def evaluate(ctx: typer.Context) -> None:
    """Evaluate (Hydra-powered)."""
    _run_hydra(["mode=evaluate", *ctx.args])


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload for development"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Path to model checkpoint (overrides config)"),
) -> None:
    """Start the API server for model serving.
    
    Uses configuration from configs/config.yaml for checkpoint paths and GCP settings.
    """
    import os

    import uvicorn

    from configs import get_config

    # Load config
    config = get_config()
    
    # Set environment variables from config for the API to use
    if checkpoint:
        # Use local checkpoint directly
        os.environ["CHECKPOINT_LOCAL"] = str(Path(checkpoint).resolve())
    else:
        # Use config values
        serving_checkpoint = config.get("checkpoint", {}).get("serving_checkpoint", "")
        if serving_checkpoint and Path(serving_checkpoint).exists():
            os.environ["CHECKPOINT_LOCAL"] = str(Path(serving_checkpoint).resolve())
    
    # Set GCP bucket/blob from config
    gcp_config = config.get("gcp", {})
    if "bucket" in gcp_config:
        os.environ.setdefault("BUCKET_NAME", gcp_config["bucket"])
    
    checkpoint_config = config.get("checkpoint", {})
    if "serving_checkpoint" in checkpoint_config:
        # Extract blob name from the checkpoint path
        os.environ.setdefault("CHECKPOINT_BLOB", checkpoint_config["serving_checkpoint"])
    
    typer.echo(f"Starting API server on {host}:{port}")
    uvicorn.run(
        "wizard_ops.backend.api:app",
        host=host,
        port=port,
        reload=reload,
    )


def main() -> None:
    app(prog_name="wizard_ops")
