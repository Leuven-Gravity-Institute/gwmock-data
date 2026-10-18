"""The stack's HDF5 files: written and read without h5py, and the same files gwmock-signal wrote."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pytest

from gwmock_data.multichannel import _HDF5_STACK_ORDER_ATTR, DetectorStrainStack
from gwmock_data.timeseries import TimeSeries

try:
    import h5py
except ModuleNotFoundError as exc:  # Python 3.15: h5py publishes no wheel yet
    if exc.name != "h5py":
        raise
    h5py = None

_T0 = 1_400_000_000.25
_NAMES = ["L1", "H1", "V1"]  # deliberately not in name order


def _stack() -> DetectorStrainStack:
    rng = np.random.default_rng(3)
    return DetectorStrainStack.from_mapping(
        _NAMES, {name: TimeSeries(rng.normal(size=64), t0=_T0, sample_rate=16.0) for name in _NAMES}
    )


def _hide(monkeypatch, package):
    real_import = importlib.import_module

    def missing(name, *args, **kwargs):
        if name == package or name.startswith(f"{package}."):
            raise ModuleNotFoundError(f"No module named {package!r}", name=package)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", missing)


def test_without_h5py_a_stack_round_trips_in_detector_order(monkeypatch, tmp_path):
    """Hidden h5py: the stack is written by the built-in writer and read back in its own order."""
    _hide(monkeypatch, "h5py")
    stack = _stack()

    stack.write(tmp_path / "stack.h5")
    back = DetectorStrainStack.read(tmp_path / "stack.h5", format="hdf5")

    assert list(back.detector_names) == _NAMES
    np.testing.assert_array_equal(back.data, stack.data)
    assert back.t0 == _T0
    assert back.sample_rate == pytest.approx(16.0, rel=1e-15)


@pytest.mark.skipif(h5py is None, reason="h5py is not installed")
def test_a_file_written_by_h5py_in_the_stack_layout_reads_without_h5py(monkeypatch, tmp_path):
    """The layout gwmock-signal wrote with h5py: datasets with t0/dt/unit and the order as a root attribute."""
    stack = _stack()
    with h5py.File(tmp_path / "legacy.h5", "w") as handle:
        handle.attrs[_HDF5_STACK_ORDER_ATTR] = json.dumps(_NAMES)
        for name in _NAMES:
            dataset = handle.create_dataset(name, data=np.asarray(stack[name].data))
            dataset.attrs.update({"t0": _T0, "dt": 1 / 16.0, "unit": "strain"})
    _hide(monkeypatch, "h5py")

    back = DetectorStrainStack.read(tmp_path / "legacy.h5", format="hdf5")

    assert list(back.detector_names) == _NAMES
    np.testing.assert_array_equal(back.data, stack.data)


def test_frames_without_gwpy_name_the_extra(monkeypatch, tmp_path):
    """GWF output without GWpy says how to install it."""
    _hide(monkeypatch, "gwpy")

    with pytest.raises(ModuleNotFoundError, match=r"pip install 'gwmock-data\[gwpy\]'"):
        _stack().write(tmp_path / "stack.gwf", format="gwf")


def test_a_group_where_a_detector_should_be_is_refused(tmp_path):
    """A group in place of a detector dataset is refused rather than read as strain."""
    from gwmock_data import hdf5

    hdf5.write(hdf5.Group({"H1": hdf5.Group()}), tmp_path / "odd.h5")

    with pytest.raises(TypeError, match="is a group"):
        DetectorStrainStack.read(tmp_path / "odd.h5", format="hdf5")
