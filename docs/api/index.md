---
title: API reference
description: Complete reference for gwmock-data.
---

This section documents the public API of `gwmock-data`.

- **[Time series](timeseries.md)** — `TimeSeries` and `as_timeseries`.
- **[Sampling grid](sampling_grid.md)** — `SamplingGrid`.
- **[Input and output](io.md)** — `read_hdf5`, `write_hdf5`.
- **[HDF5](hdf5.md)** — `Group`, `Dataset`, `read`, `write`: HDF5 files with or
  without h5py.
- **[Multichannel stack](multichannel.md)** — `DetectorStrainStack`: aligned
  strains for a fixed list of detectors.
- **[Multichannel time series](multichannel_series.md)** —
  `MultichannelTimeSeries`: several channels on one grid, injected into segment
  by segment, and `MultichannelTimeSeriesList`, an ordered list of them.
- **[Injection](injection.md)** — adding one series into another, with
  sub-sample offsets resampled by the **[shared kernel](resampling.md)**.
