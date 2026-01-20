# wizard_ops/cli.py
from __future__ import annotations

import sys

import typer

from wizard_ops import get_version
from wizard_ops.hydra_app import main as hydra_main

app = typer.Typer(add_completion=False, help="wizard_ops command-line interface")


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


def _split_on_double_dash(args: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in args:
        return args, []
    i = args.index("--")
    return args[:i], args[i + 1 :]


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


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def data(ctx: typer.Context) -> None:
    """
    Data utilities.

    Convention:
      - Hydra args first
      - then `--`
      - then data-subcommand args (parsed by Typer inside run_data_cli)
    """
    hydra_args, data_args = _split_on_double_dash(list(ctx.args))

    # Pass data args to the data CLI runner via a simple global.
    # (You can also store them in a module-level variable or similar.)
    from wizard_ops.data_cli import set_data_argv

    set_data_argv(data_args)

    _run_hydra(["mode=data", *hydra_args])


def main() -> None:
    app(prog_name="wizard_ops")
