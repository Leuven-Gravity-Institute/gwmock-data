"""Top-level package for gwmock_data."""

from __future__ import annotations

# PEP 810: defer importlib.metadata (loaded by version.py) on 3.15+.
# Ignored on 3.13/3.14 where the lazy import machinery does not exist.
__lazy_modules__ = ["gwmock_data.sampling_grid", "gwmock_data.timeseries", "gwmock_data.version"]

from gwmock_data.sampling_grid import SamplingGrid
from gwmock_data.timeseries import TimeSeries, as_timeseries
from gwmock_data.version import __version__

__all__ = ["SamplingGrid", "TimeSeries", "__version__", "as_timeseries"]
