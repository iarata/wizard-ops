"""wizard_ops package.

Keep this module lightweight: importing `wizard_ops` should not trigger heavy
ML imports or execute CLI/training code.
"""

from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    try:
        return version("wizard_ops")
    except PackageNotFoundError:
        return "0.0.0"


__all__ = ["get_version"]