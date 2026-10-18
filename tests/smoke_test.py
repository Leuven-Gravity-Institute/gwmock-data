"""Smoke tests before publishing to verify the wheel and source distribution."""

from __future__ import annotations

import sys

import gwmock_data


def test_basic_import() -> None:
    """Test basic import."""
    print(f"Python version: {sys.version}")
    print(f"Package version: {gwmock_data.__version__}")

    # Ensure it's not importing the local folder
    if "site-packages" not in gwmock_data.__file__ and "dist" not in gwmock_data.__file__:
        print(f"Warning: Package imported from unexpected location: {gwmock_data.__file__}")


if __name__ == "__main__":
    test_basic_import()
    print("Smoke test passed: Package is importable.")
