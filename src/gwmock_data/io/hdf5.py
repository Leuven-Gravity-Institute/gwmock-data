"""Read and write time series as HDF5, in the layout GWpy's ``TimeSeriesDict`` uses.

One dataset per series, named by its key and gzip-compressed, with the attributes GWpy writes and
reads: ``x0`` (the GPS time of the first sample), ``dx`` (the sample spacing, in seconds),
``xunit`` (``"s"``), ``unit`` and, for a series that has one, ``name``. A file written here therefore loads with
``gwpy.timeseries.TimeSeriesDict.read(path, format="hdf5")``, and a file GWpy wrote loads here.

Neither h5py nor GWpy is needed: files go through :mod:`gwmock_data.hdf5`, which uses h5py when it is installed
(and then gzip-compresses the datasets, as GWpy does) and its own writer and pyfive otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from gwmock_data import hdf5
from gwmock_data.timeseries import TimeSeries, as_timeseries

__all__ = ["read_hdf5", "write_hdf5"]


def write_hdf5(series: Mapping[str, TimeSeries], path: str | Path) -> None:
    """Write each series to ``path`` as one HDF5 dataset named by its key, replacing the file.

    Args:
        series: Mapping of dataset name to a one-dimensional time series (this package's or
            GWpy's).
        path: Output file.

    Raises:
        ValueError: If a series is not one-dimensional.
    """
    channels = {key: as_timeseries(value) for key, value in series.items()}
    for key, channel in channels.items():
        if len(channel.data.shape) != 1:
            raise ValueError(f"{key!r} must be one-dimensional to write; it has shape {tuple(channel.data.shape)}.")
    members: dict[str, hdf5.Group | hdf5.Dataset] = {}
    for key, channel in channels.items():
        attrs = {"x0": channel.t0, "dx": channel.dt, "xunit": "s", "unit": channel.unit or ""}
        if channel.name is not None:
            attrs["name"] = channel.name
        members[key] = hdf5.Dataset(np.asarray(channel.data), attrs)
    hdf5.write(hdf5.Group(members), Path(path), compression="gzip")


def read_hdf5(path: str | Path) -> dict[str, TimeSeries]:
    """Read every dataset in ``path`` written in the layout :func:`write_hdf5` writes.

    Args:
        path: Input file.

    Returns:
        Mapping of dataset name to :class:`~gwmock_data.timeseries.TimeSeries`, in file order.

    Raises:
        TypeError: If a member is a group rather than a dataset.
        ValueError: If a dataset lacks the ``x0`` or ``dx`` attribute, or its time unit is not seconds.
    """
    result = {}
    for key, dataset in hdf5.read(Path(path)).members.items():
        if isinstance(dataset, hdf5.Group):
            raise TypeError(f"{key!r} is a group, not a time series dataset.")
        attrs = dataset.attrs
        if "x0" not in attrs or "dx" not in attrs:
            raise ValueError(f"dataset {key!r} has no x0 and dx attributes, so it is not a time series.")
        xunit = _text(attrs.get("xunit", "s"))
        if xunit not in ("s", "second"):
            raise ValueError(f"dataset {key!r} has a time axis in {xunit!r}; only seconds are supported.")
        name = _text(attrs["name"]) if "name" in attrs else None
        result[key] = TimeSeries(
            np.asarray(dataset.data),
            t0=float(attrs["x0"]),
            sample_rate=1.0 / float(attrs["dx"]),
            name=name,
            unit=_text(attrs.get("unit", "")) or None,
        )
    return result


def _text(value: object) -> str:
    """Return an HDF5 string attribute as ``str``, whether h5py read it as bytes or text."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)
