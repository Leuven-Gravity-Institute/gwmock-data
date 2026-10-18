"""The stack takes the package's own time series as well as GWpy's.

Native and GWpy inputs give identical results, and what comes back is the package's series. A GWpy
series given is converted once, sharing its samples.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_data.multichannel import DetectorStrainStack
from gwmock_data.timeseries import TimeSeries

try:
    import gwpy  # noqa: F401
except ModuleNotFoundError as exc:  # Python 3.15: GWpy needs h5py, which publishes no wheel yet
    if exc.name != "gwpy":
        raise
    _HAVE_GWPY = False
else:
    _HAVE_GWPY = True

_needs_gwpy = pytest.mark.skipif(not _HAVE_GWPY, reason="GWpy is not installed")

_FS = 64.0
_T0 = 1_400_000_000.0


def _native(value: float, *, n: int = 32, t0: float = _T0, **kwargs: object) -> TimeSeries:
    return TimeSeries(np.full(n, value), t0=t0, sample_rate=_FS, **kwargs)


class TestStack:
    """Channels of either kind; read back as the package's series."""

    def test_native_channels_are_returned_as_given(self):
        """The same object every time, with the channel's samples and timing."""
        given = _native(1.0)
        stack = DetectorStrainStack.from_mapping(["H1", "L1"], {"H1": given, "L1": _native(2.0)})

        assert stack["H1"] is given
        assert stack[0] is given
        assert stack.to_dict()["H1"] is given
        assert stack.t0 == _T0
        assert stack.sample_rate == _FS
        assert np.array_equal(stack.data, np.stack([np.full(32, 1.0), np.full(32, 2.0)]))

    @_needs_gwpy
    def test_a_gwpy_channel_is_returned_as_the_package_s_series_sharing_its_samples(self):
        """Converted once at construction; the same object on every read, over GWpy's array."""
        given = _native(2.0).to_gwpy()
        stack = DetectorStrainStack.from_mapping(["H1", "L1"], {"H1": _native(1.0), "L1": given})

        assert isinstance(stack["L1"], TimeSeries)
        assert stack["L1"] is stack["L1"]
        assert np.shares_memory(stack["L1"].data, given.value)

    @pytest.mark.parametrize(
        ("other", "match"),
        [
            (_native(1.0, n=16), "length 16 does not match"),
            (_native(1.0, t0=_T0 + 1.0), "time grid does not match"),
            (_native(1.0, unit="strain"), "not compatible"),
            (TimeSeries(np.zeros((2, 32)), t0=_T0, sample_rate=_FS), "must be one-dimensional"),
        ],
    )
    def test_misaligned_native_channels_are_rejected(self, other, match):
        """Length, start, unit and shape are checked on native channels as on GWpy ones."""
        with pytest.raises(ValueError, match=match):
            DetectorStrainStack.from_mapping(["H1", "L1"], {"H1": _native(1.0), "L1": other})

    def test_anything_else_is_rejected_naming_the_detector(self):
        """A bare array has no timing."""
        with pytest.raises(TypeError, match=r"strains\['L1'\] must be a gwmock_data.TimeSeries"):
            DetectorStrainStack.from_mapping(["H1", "L1"], {"H1": _native(1.0), "L1": np.zeros(32)})

    @pytest.mark.parametrize("rate", [_FS, 7.5, 3000.0])
    @pytest.mark.parametrize("fmt", ["hdf5", "npy"])
    def test_a_native_stack_survives_a_file_round_trip(self, tmp_path, fmt, rate):
        """Samples, start time and rate come back exactly, at rates that are not whole or dyadic too."""
        channels = {
            name: TimeSeries(np.full(32, value), t0=_T0, sample_rate=rate) for name, value in (("H1", 1.0), ("L1", 2.0))
        }
        stack = DetectorStrainStack.from_mapping(["H1", "L1"], channels)
        path = tmp_path / f"stack.{fmt}"

        stack.write(path, format=fmt)
        back = DetectorStrainStack.read(path, format=fmt)

        assert back.detector_names == ("H1", "L1")
        assert np.array_equal(back.data, stack.data)
        assert back.t0 == _T0
        assert back.sample_rate == rate


class TestOneSetOfSamples:
    """A stack's channel, its ``data`` and the caller's series stay one set of samples."""

    @_needs_gwpy
    def test_a_gwpy_channel_and_the_stack_share_samples(self):
        """A write through the GWpy series given is seen by the stack's channel and ``data``."""
        given = _native(1.0).to_gwpy()
        stack = DetectorStrainStack.from_mapping(["H1"], {"H1": given})

        given.value[0] = 7.0
        stack["H1"].data[1] = 8.0

        assert stack.data[0, :2].tolist() == [7.0, 8.0]
        assert given.value[:2].tolist() == [7.0, 8.0]

    @_needs_gwpy
    def test_the_timing_is_taken_when_the_stack_is_built(self, tmp_path):
        """Moving a GWpy series given afterwards changes neither the stack nor what it writes."""
        given = _native(1.0).to_gwpy()
        stack = DetectorStrainStack.from_mapping(["H1"], {"H1": given})
        given.t0 = _T0 + 5.0

        stack.write(tmp_path / "stack.npy", format="npy")

        assert stack.t0 == _T0
        assert DetectorStrainStack.read(tmp_path / "stack.npy", format="npy").t0 == _T0

    @_needs_gwpy
    def test_frames_record_the_stack_s_timing_and_names(self, monkeypatch, tmp_path):
        """GWF output carries the stack's start time, each channel named for its detector."""
        from gwpy.timeseries import TimeSeriesDict

        written = {}
        monkeypatch.setattr(TimeSeriesDict, "write", lambda self, *args, **kwargs: written.update(self))
        stack = DetectorStrainStack.from_mapping(["H1", "L1"], {"H1": _native(1.0), "L1": _native(2.0)})

        stack.write(tmp_path / "stack.gwf", format="gwf")

        assert [written[name].t0.value for name in ("H1", "L1")] == [_T0, _T0]
        assert [written[name].name for name in ("H1", "L1")] == ["H1", "L1"]
        assert np.array_equal(written["L1"].value, np.full(32, 2.0))
