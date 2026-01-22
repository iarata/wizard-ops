# wizard_ops/cli.py
from __future__ import annotations

import sys

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


def main() -> None:
    app(prog_name="wizard_ops")
