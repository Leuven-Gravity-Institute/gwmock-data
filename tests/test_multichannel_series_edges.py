"""``MultichannelTimeSeries`` branches the segment loop reaches: refusals, chunks outside, gaps, checkpoints."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from gwmock_data.injection import alignment_tolerance
from gwmock_data.multichannel_series import MultichannelTimeSeries, MultichannelTimeSeriesList

_EPOCH = 1_400_000_000.0
_FS = 16.0


def _series(start: float, samples: int, *, rate: float = _FS, channels: int = 2, value: float = 1.0):
    return MultichannelTimeSeries(np.full((channels, samples), value), start_time=start, sampling_frequency=rate)


def test_the_shape_is_channels_by_samples():
    """``shape`` is ``(num_of_channels, n_samples)``."""
    assert _series(_EPOCH, 32, channels=3).shape == (3, 32)


def test_a_chunk_at_another_sample_rate_is_refused():
    """Grids must match; a different rate would need resampling the segment loop never asked for."""
    with pytest.raises(ValueError, match="must match sampling frequency of segment"):
        _series(_EPOCH, 32).inject(_series(_EPOCH, 8, rate=32.0))


def test_a_chunk_entirely_after_the_segment_comes_back_untouched(caplog):
    """Nothing is placed, the chunk is returned for a later segment, and the reason is logged."""
    segment = _series(_EPOCH, 32, value=0.0)
    chunk = _series(_EPOCH + 10.0, 8)

    with caplog.at_level(logging.WARNING, logger="gwmock_data"):
        remaining = segment.inject(chunk)

    assert remaining is chunk
    np.testing.assert_array_equal(np.asarray(segment[0].data), np.zeros(32))
    assert "starts after the current time series ends" in caplog.text


def test_content_lost_into_a_configured_gap_is_not_reported_as_a_placement_failure(caplog):
    """A chunk reaching back into a configured gap logs at debug, not the placement-failure warning."""
    segment = _series(_EPOCH, 32, value=0.0)
    chunk = _series(_EPOCH - 1.0, 32)

    with caplog.at_level(logging.DEBUG, logger="gwmock_data"):
        segment.inject(chunk, preceding_gap=(_EPOCH - 2.0, _EPOCH))

    assert "into the configured gap" in caplog.text
    assert not [
        record for record in caplog.records if record.levelno >= logging.WARNING and "Discarding" in record.message
    ]


@pytest.mark.parametrize(("offset", "expected"), [(0.5, True), (1.9, True), (1.99, False), (2.0, False), (-0.6, False)])
def test_contributes_samples_follows_the_grid(offset, expected):
    """A chunk contributes exactly when a grid point of the segment lies within its sample span."""
    segment = _series(_EPOCH, 32)  # covers [epoch, epoch + 2) s at 16 Hz
    chunk = _series(_EPOCH + offset, 8)  # 0.5 s long

    assert segment.contributes_samples(chunk) is expected


def test_inject_from_list_returns_the_tails():
    """Each chunk is placed; what runs past the segment's end comes back, in a list, for the next segment."""
    segment = _series(_EPOCH, 32, value=0.0)
    inside, across = _series(_EPOCH + 0.5, 8), _series(_EPOCH + 1.75, 16)

    remaining = segment.inject_from_list([inside, across])

    assert isinstance(remaining, MultichannelTimeSeriesList)
    assert len(remaining) == 1
    assert float(remaining[0].start_time.value) == _EPOCH + 2.0
    assert remaining[0].shape == (2, 12)


def test_a_checkpoint_with_more_channel_records_than_channels_restores():
    """Extra channel records are ignored rather than raising, so an odd checkpoint still resumes."""
    record = _series(_EPOCH, 8).to_json_dict()
    record["channels"] = [*record["channels"], {"name": "extra", "channel": "X1:EXTRA", "unit": ""}]

    restored = MultichannelTimeSeries.from_json_dict(record)

    assert restored.num_of_channels == 2


def test_an_empty_list_of_times_leaves_the_tolerance_at_the_epoch():
    """No extra timestamps: the tolerance is the one at the reference time alone."""
    assert alignment_tolerance(_EPOCH, _FS, gps_times=[]) == alignment_tolerance(_EPOCH, _FS)
