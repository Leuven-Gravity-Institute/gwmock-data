"""Test importing the gwmock_data package and its modules."""

from __future__ import annotations

import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import gwmock_data


def get_all_submodules(package):
    """Discover all submodules in the package.

    Args:
        package: The package to inspect.

    """
    submodules = []
    for _, mod_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        submodules.append(mod_name)
    return submodules


def test_import_main_package():
    """Test that the main gwmock_data package can be imported."""
    assert hasattr(gwmock_data, "__version__")
    assert gwmock_data.__version__ is not None


@pytest.mark.parametrize("module_name", get_all_submodules(gwmock_data))
def test_import_submodule(module_name):
    """Test that all submodules can be imported."""
    __import__(module_name)


@pytest.mark.skipif(sys.version_info < (3, 15), reason="lazy imports (PEP 810) need Python 3.15+")
def test_lazy_import_defers_importlib_metadata():
    """Importing the package must not load importlib.metadata until __version__ is read.

    Runs in a fresh interpreter, because the test session has already imported the package.
    Variables that force eager imports or start coverage in subprocesses are removed so they
    cannot load importlib.metadata early.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COV_", "COVERAGE_", "PYTHON_LAZY_"))}
    code = (
        "import sys; sys.path.insert(0, 'src'); "
        "import gwmock_data as p; "
        "before = 'importlib.metadata' in sys.modules; "
        "p.__version__; "
        "print(before, 'importlib.metadata' in sys.modules)"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        check=True,
    )
    loaded_at_import, loaded_after_version = result.stdout.split()
    assert loaded_at_import == "False", "importlib.metadata was loaded while importing the package"
    assert loaded_after_version == "True", "reading __version__ did not load importlib.metadata"
