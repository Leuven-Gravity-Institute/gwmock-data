"""The stack's refusals and fallbacks: what it rejects, and how it reads a file with an unusable order."""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_data import hdf5
from gwmock_data.multichannel import _HDF5_STACK_ORDER_ATTR, DetectorStrainStack
from gwmock_data.timeseries import TimeSeries


def _series(value: float = 0.0) -> TimeSeries:
    return TimeSeries(np.full(4, value), t0=10.0, sample_rate=4.0)


@pytest.mark.parametrize(
    ("names", "channels", "error", "match"),
    [
        (("H1", "L1"), (_series(),), ValueError, "same length"),
        (("H1", "H1"), (_series(), _series()), ValueError, "unique"),
        ((), (), ValueError, "At least one channel"),
        (("H1",), ([0.0, 1.0],), TypeError, r"channels\[0\] must be a gwmock_data.TimeSeries"),
    ],
)
def test_the_constructor_refuses_inconsistent_input(names, channels, error, match):
    """Each check names what is wrong rather than failing later on a mismatch."""
    with pytest.raises(error, match=match):
        DetectorStrainStack(names, channels)


def test_indexing_by_position_name_and_length():
    """Integers and names reach the same channels; booleans and other keys are refused."""
    stack = DetectorStrainStack(("H1", "L1"), (_series(1.0), _series(2.0)))

    assert len(stack) == 2
    assert stack[1] is stack["L1"]
    with pytest.raises(TypeError, match="got bool"):
        stack[True]
    with pytest.raises(TypeError, match="index must be int or str"):
        stack[1.0]


@pytest.mark.parametrize(
    "order", ["not json", '"H1"', '["H1"]', '["H1", "H1"]', '["H1", "X1"]', "[1, 2]"], ids=lambda value: value
)
def test_an_unusable_order_attribute_falls_back_to_the_file_order(tmp_path, order):
    """Malformed, partial, repeated or foreign names in the order attribute are ignored, not trusted."""
    members = {
        name: hdf5.Dataset(np.full(4, value), {"t0": 10.0, "dt": 0.25, "unit": "strain"})
        for name, value in (("L1", 2.0), ("H1", 1.0))
    }
    hdf5.write(hdf5.Group(members, {_HDF5_STACK_ORDER_ATTR: order}), tmp_path / "stack.h5")

    stack = DetectorStrainStack.read(tmp_path / "stack.h5", format="hdf5")

    assert stack.detector_names == ("H1", "L1")
    np.testing.assert_array_equal(stack["L1"].data, np.full(4, 2.0))


def test_reading_an_unknown_format_is_refused(tmp_path):
    """Only the formats the stack writes and can read back are accepted."""
    with pytest.raises(ValueError, match="Unknown format"):
        DetectorStrainStack.read(tmp_path / "stack.csv", format="csv")  # type: ignore[arg-type]
