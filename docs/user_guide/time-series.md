---
title: Time series
description:
    The time series the gwmock packages share, and how to move code written
    against GWpy series.
---

[`gwmock_data.TimeSeries`](../api/timeseries.md) is the series the gwmock
packages pass between each other: an array of samples placed in time by the GPS
time of its first sample and its sample rate.

```python
import numpy as np

from gwmock_data import TimeSeries

series = TimeSeries(np.zeros(4096), t0=1_400_000_000.0, sample_rate=4096.0, name="H1", unit="strain")

series.t0  # 1400000000.0, GPS seconds (a float)
series.sample_rate  # 4096.0, Hz
series.dt  # 1 / 4096, seconds
series.duration  # 1.0, seconds
series.n_samples  # 4096
series.times  # GPS time of each sample
series.data  # the samples; `series.value` is the same array
```

- **Samples.** `data` is a NumPy or a JAX array, kept as given. Time runs along
  the **last** axis; leading axes share the timing, so the strain of several
  detectors over one segment can be one series of shape
  `(n_detectors, n_samples)`.
- **Timing.** `t0` and `sample_rate` are plain floats. Sample `k` is at
  `t0 + k * dt`, the same lattice [`SamplingGrid`](../api/sampling_grid.md)
  uses.
- **New series.** `crop(start, end)` (GPS seconds), `pad(before, after)`
  (samples), `copy()` and `with_data(data)` return new series; a series is never
  changed in place by them.
- **JAX.** A JAX-backed series stays on its device. JAX must be in 64-bit mode
  (`jax.config.update("jax_enable_x64", True)`): GPS times need double
  precision, and the methods that would lose it raise instead.

## Inputs

`gwmock_data.as_timeseries` turns a [GWpy](https://gwpy.github.io) `TimeSeries`
into this package's, without copying its samples, and returns this package's
series unchanged. A function that takes a time series can call it on its input
to accept either kind.

## Converting

| To                 | Call                                           | Needs               |
| ------------------ | ---------------------------------------------- | ------------------- |
| GWpy `TimeSeries`  | `series.to_gwpy()`                             | the `[gwpy]` extra  |
| from GWpy          | `TimeSeries.from_gwpy(gwpy_ts)`                | the `[gwpy]` extra  |
| PyCBC `TimeSeries` | `series.to_pycbc()`                            | the `[pycbc]` extra |
| from PyCBC         | `TimeSeries.from_pycbc(pycbc_ts)`              | the `[pycbc]` extra |
| NumPy samples      | `series.to_numpy()` or `numpy.asarray(series)` |                     |
| JAX samples        | `series.to_jax()`                              | the `[jax]` extra   |

`to_gwpy()` and `from_gwpy()` share NumPy samples rather than copying them.

## Files

`gwmock_data.io.write_hdf5` and `read_hdf5` write and read a mapping of series
as HDF5 in the layout GWpy's `TimeSeriesDict` uses, so either side can read what
the other wrote. They need neither h5py nor GWpy: without h5py, files are
written by gwmock-data's own HDF5 writer and read with pyfive, a pure-Python
HDF5 reader, so HDF5 works on Python 3.15 too. With h5py installed (the `[hdf5]`
extra) it is used instead, and compresses the datasets as GWpy does.

[`gwmock_data.hdf5`](../api/hdf5.md) is the layer underneath, for any HDF5 file:
`write(Group(...), path)` and `read(path)` work on a tree of groups and datasets
with attributes.

```python
from gwmock_data.io import read_hdf5, write_hdf5

write_hdf5({"H1": series}, "strain.h5")
again = read_hdf5("strain.h5")  # {"H1": TimeSeries(...)}
```

## Moving from GWpy series

Code written against GWpy's `TimeSeries` changes as follows:

| GWpy                           | `gwmock_data.TimeSeries`                   |
| ------------------------------ | ------------------------------------------ |
| `ts.value`                     | `ts.value` or `ts.data`                    |
| `ts.t0.value`                  | `ts.t0`                                    |
| `ts.sample_rate.value`         | `ts.sample_rate`                           |
| `ts.dt.value`                  | `ts.dt`                                    |
| `ts.duration.value`            | `ts.duration`                              |
| `ts.times.value`               | `ts.times`                                 |
| `len(ts)`                      | `ts.n_samples`                             |
| `ts.unit` (an astropy unit)    | `ts.unit` (a string, or `None`)            |
| `TimeSeries(x, t0=t0, dt=dt)`  | `TimeSeries(x, t0=t0, sample_rate=1 / dt)` |
| arithmetic, `fft`, `plot`, ... | `ts.to_gwpy()` first                       |

Code that needs a GWpy object can convert any series with `to_gwpy()`.
