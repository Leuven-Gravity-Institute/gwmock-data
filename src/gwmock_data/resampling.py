"""The band-limited resampling kernel: a Kaiser-windowed sinc.

Shifting a uniformly sampled, band-limited signal by a fraction of a sample needs its values between samples. The
*exact* interpolant is the sinc series; truncating it to a finite window is the only approximation, and its error
falls steeply with the number of taps. That makes accuracy a tunable rather than a fixed property of the method,
unlike a cubic, whose error is fixed at O(h^4).

The gwmock packages resample with this one kernel -- gwmock-signal when it projects onto a detector and places a
signal on the output lattice, gwmock when it injects a chunk that sits off a segment's grid -- so they agree on the
same operation. The specification (tap count, window shape, weight formula) is defined here once.

!!! note "Choosing the tap count"

    The defaults were set by measured convergence against an interpolation-free direct-Fourier reference, not
    from a design formula. A Kaiser window is used because its stopband is controlled by a single parameter, so
    tap count and attenuation can be traded explicitly.

    No windowed-sinc kernel resamples accurately for content arbitrarily close to Nyquist: the default is accurate
    to 0.8 x Nyquist and fails above it (about 0.4 of a tone's amplitude at 0.98 x Nyquist). A series with content up
    to its Nyquist frequency is therefore oversampled first, exactly, with :func:`oversample_fourier`, which puts
    the whole of its band at or below half the kernel's Nyquist frequency.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

#: Fewest taps that still defines a symmetric kernel.
_MINIMUM_TAPS = 3

#: Default number of kernel taps. Odd, so the kernel is symmetric about the sample
#: preceding the requested position.
DEFAULT_SINC_TAPS = 127

#: Default Kaiser window shape parameter. This, not the tap count, sets the error floor:
#: at 127 taps, beta = 12 stalls near 1e-8 while beta = 32 reaches 1e-12. It cannot be
#: raised freely, because a larger beta needs more taps to hold its transition band --
#: see :data:`_TAPS_PER_BETA` and :func:`validate_kernel`.
DEFAULT_KAISER_BETA = 32.0

#: Output positions evaluated per pass of the tap loop. The loop builds about a dozen temporaries the
#: size of what it evaluates, once per tap, so evaluating a whole buffer at once costs memory in
#: proportion to the buffer and works out of main memory rather than cache. Measured at 4e6 positions,
#: blocks of this size cut the call's peak allocation from 0.72 GB to 0.044 GB and ran 1.9x faster.
#: Every operation in the loop is elementwise, so the result does not depend on the block size, bit
#: for bit.
_BLOCK_POSITIONS = 65536

#: A whole sample or a fraction beyond this many samples from the start is taken as outside any series
#: (none fits in memory), so whole samples and carried fractions are added only below it and cannot overflow.
_FAR = 2**61

#: Taps required per unit of Kaiser beta. Measured, not derived: at beta = 32, 127 taps
#: reach 2.2e-12 at 0.8 x Nyquist while 63 taps give only 1.7e-4, because the transition
#: band no longer fits. The bound ``taps >= 4 * beta - 1`` reproduces where that
#: transition happens across the beta values tested.
_TAPS_PER_BETA = 4.0


def validate_kernel(taps: int, beta: float) -> tuple[int, float]:
    """Validate a kernel specification and return it normalised.

    Args:
        taps: Number of kernel taps; must be an odd integer >= 3.
        beta: Kaiser window shape parameter; must be finite, non-negative and small enough
            that its transition band fits in ``taps``.

    Returns:
        ``(taps, beta)`` as an ``int`` and a ``float``.

    Raises:
        ValueError: If ``taps`` is not an odd integer >= 3, if ``beta`` is not finite or is
            negative, or if ``beta`` is too large for ``taps``. The last case is rejected
            rather than warned about because it fails *quietly*: the kernel still returns
            plausible numbers, several orders of magnitude less accurate than the tap count
            implies.
    """
    # Rejected rather than truncated: int(127.9) would silently accept a value the
    # integer-only contract forbids, and the caller would never learn which kernel ran.
    if isinstance(taps, float) and not taps.is_integer():
        raise ValueError(f"taps must be an odd integer >= 3; got {taps}.")
    taps = int(taps)
    if taps < _MINIMUM_TAPS or taps % 2 == 0:
        raise ValueError(f"taps must be an odd integer >= 3; got {taps}.")
    beta = float(beta)
    # NaN fails every comparison, so it would slip past a bare `beta < 0` check and then
    # propagate silently through every interpolated sample.
    if not np.isfinite(beta):
        raise ValueError(f"Kaiser beta must be finite; got {beta}.")
    if beta < 0.0:
        raise ValueError(f"Kaiser beta must be non-negative; got {beta}.")
    minimum_taps = _TAPS_PER_BETA * beta - 1.0
    if taps < minimum_taps:
        raise ValueError(
            f"Kaiser beta={beta} needs at least {minimum_taps:.0f} taps to hold its transition "
            f"band, but taps={taps}. Either raise taps or lower beta; a beta too large for the "
            f"kernel silently loses several orders of magnitude of accuracy."
        )
    return taps, beta


#: Factor by which :func:`oversample_fourier` refines a series before the kernel reads it. Twice the rate puts
#: the input's whole band, up to its own Nyquist frequency, at or below half the kernel's Nyquist frequency,
#: well inside the 0.8 x Nyquist the default kernel is accurate to; a larger factor gains nothing there and
#: costs memory in proportion.
OVERSAMPLING_FACTOR = 2


def oversample_fourier(samples: np.ndarray, factor: int = OVERSAMPLING_FACTOR) -> np.ndarray:
    """Return the band-limited interpolant of *samples* on a grid *factor* times finer.

    Sample ``k`` of *samples* is sample ``factor * k`` of the result, which has ``factor * (n - 1) + 1`` samples,
    so a position ``p`` on the input's grid is the position ``factor * p`` on the result's.

    Computed in the frequency domain: the spectrum is extended with zeros up to the finer grid's Nyquist
    frequency, so nothing is added or removed below the input's own. Before the transform the input is padded
    with zeros to twice its length, so the circular transform does not wrap one end of the series into the
    other. A component at exactly the input's Nyquist frequency has no unique interpolant (its samples are
    those of a cosine at that frequency whatever its phase); it is split evenly between the positive and the
    negative frequency, which keeps the result real and reproduces the input's samples.

    The interpolant is global, as it has to be to stay accurate up to Nyquist: a series is interpolated as one
    that is zero outside its samples, so one that ends abruptly rings near its ends with a tail falling as one over
    the distance, where the windowed-sinc kernel alone is affected only within half its taps. A waveform tapered to
    zero at both ends, as generated signals are, is unaffected. LAL's injection
    (``XLALSimAddInjectionREAL8TimeSeries``) shifts by a fraction of a sample the same way, in the frequency domain
    over the whole zero-padded series.

    Peak memory is about 110 bytes per input sample at a factor of 2 (6.1 GB at 5.2e7 samples, a 1.8-hour series
    at 8192 Hz), most of it the transforms' own working arrays.

    Args:
        samples: Uniformly sampled series, shape ``(n,)``.
        factor: How many times finer the result's grid is; a positive integer.

    Returns:
        The interpolated series, shape ``(factor * (n - 1) + 1,)``; empty for an empty input.

    Raises:
        ValueError: If ``factor`` is not a positive integer, or ``samples`` is not one-dimensional.
    """
    if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError(f"factor must be a positive integer; got {factor!r}.")
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1:
        raise ValueError(f"samples must be one-dimensional; got shape {samples.shape}.")
    n = samples.shape[0]
    if n == 0 or factor == 1:
        return samples.copy()
    padded = 2 * n
    spectrum = np.fft.rfft(samples, padded)
    spectrum[-1] *= 0.5
    # `irfft` extends the spectrum with zeros to the requested length itself, so the finer spectrum is
    # never built as a separate array.
    return np.fft.irfft(spectrum, factor * padded)[: factor * (n - 1) + 1] * factor


def resample_uniform_sinc(
    samples: np.ndarray,
    index: np.ndarray,
    *,
    fraction: np.ndarray | float | None = None,
    taps: int = DEFAULT_SINC_TAPS,
    beta: float = DEFAULT_KAISER_BETA,
) -> np.ndarray:
    """Resample a uniformly sampled series at fractional positions (NumPy).

    Evaluates the Kaiser-windowed sinc series. Positions outside
    ``[0, len(samples) - 1]`` return zero, matching the ``fill_value=0.0`` behaviour of
    the interpolation this replaces.

    Weights are normalised to sum to one, which enforces unit gain at DC and removes
    the amplitude bias the window would otherwise introduce near the ends of the series.

    A position given as one float64 keeps about 16 significant digits, so its fractional part is
    resolved only to ``index * 2.2e-16`` of a sample: about 1e-9 of a sample at ten million samples,
    which the kernel turns into an error of that much time times the signal's slope. A caller whose
    positions are a large whole number plus a small offset should pass them as such: ``index`` the
    integer part and ``fraction`` the offset. The two are never added in floating point, so the
    offset keeps its own precision whatever the size of ``index``.

    Args:
        samples: Uniformly sampled series, shape ``(n,)``.
        index: Sample positions to evaluate at, any shape: fractional, or, with ``fraction``, whole
            numbers (an integer array).
        fraction: Offset added to each position of ``index``, in samples; a scalar or an array
            broadcastable to ``index``. Any size, though its precision is that of its own magnitude.
        taps: Number of kernel taps.
        beta: Kaiser window shape parameter.

    Returns:
        Interpolated values, the shape of ``index``.

    Raises:
        TypeError: If ``fraction`` is given and ``index`` is not an integer array castable to ``int64``
            (``uint64`` is not: its values above ``2**63`` would wrap).
    """
    taps, beta = validate_kernel(taps, beta)
    samples = np.asarray(samples, dtype=float)
    if fraction is None:
        positions = np.asarray(index, dtype=float)
        whole = np.floor(positions)
        part = positions - whole
        whole = whole.astype(np.int64)
    else:
        whole = np.asarray(index)
        if not np.issubdtype(whole.dtype, np.integer) or not np.can_cast(whole.dtype, np.int64):
            raise TypeError(
                f"index must be an integer array castable to int64 when fraction is given, not {whole.dtype}"
            )
        offset = np.broadcast_to(np.asarray(fraction, dtype=float), whole.shape)
        whole = whole.astype(np.int64)
        carried = np.floor(np.where(np.isfinite(offset), offset, 0.0))
        # A whole sample or a carried fraction this far out puts the position outside any series, and so
        # does a non-finite fraction (as a non-finite float position): such a position places nothing.
        # Excluding them first means the sum below cannot overflow.
        far = ~np.isfinite(offset) | (whole > _FAR) | (whole < -_FAR) | (np.abs(carried) > _FAR)
        whole = np.where(far, -1, whole + np.where(far, 0.0, carried).astype(np.int64))
        part = np.where(far, 0.0, offset - carried)
    shape = whole.shape
    whole, part = whole.ravel(), part.ravel()
    result = np.empty(whole.shape, dtype=float)
    for first in range(0, whole.size, _BLOCK_POSITIONS):
        block = slice(first, first + _BLOCK_POSITIONS)
        result[block] = _resample_block(samples, whole[block], part[block], taps=taps, beta=beta)
    return result.reshape(shape)


def _resample_block(
    samples: np.ndarray, base_int: np.ndarray, frac: np.ndarray, *, taps: int, beta: float
) -> np.ndarray:
    """Evaluate :func:`resample_uniform_sinc` at one block of positions.

    Each position is the whole sample ``base_int`` plus ``frac`` in ``[0, 1)``; the kernel is validated.
    """
    n = samples.shape[0]
    half = (taps - 1) // 2

    # One sine for the whole tap loop. Every tap argument is `x = frac - offset` for an integer
    # offset, and `sin(pi*(frac - offset)) = (-1)**offset * sin(pi*frac)` is an identity in the
    # reals, so `taps` transcendentals collapse to one and each tap keeps only a sign flip --
    # which is exact in IEEE 754. Reduced to [-1/2, 1/2] rather than left in [0, 1): both cost
    # one sine, but `frac` near 1 puts the argument next to pi, where rounding the product
    # `fl(pi)*frac` costs the result its leading digits, and the reduced form never goes there.
    # `nearest` is 0 or 1, so the subtraction is exact (Sterbenz above 1/2, trivially below).
    #
    # The `(-1)**nearest` factor is **not load-bearing**: it is one sign for the whole position,
    # so it negates `total` and `weight_sum` alike and cancels exactly in their quotient --
    # measured bit-identical with and without it. It is kept so that each `weight` below is the
    # kernel weight itself rather than its negation, which is what the normalisation, the
    # `weight_sum != 0` test and any future reader are entitled to assume. A reader who takes it
    # for a correctness guard would be wrong; a reader who removes it makes the result depend on
    # the normalisation staying.
    nearest = np.round(frac)
    hoisted_sine = np.where(nearest == 0.0, 1.0, -1.0) * np.sin(np.pi * (frac - nearest))

    total = np.zeros(frac.shape, dtype=float)
    weight_sum = np.zeros(frac.shape, dtype=float)
    # Kaiser window evaluated on the kernel's own support, so the taper is a property
    # of the kernel rather than of where in the series it happens to land.
    denominator = half + 1.0
    for offset in range(-half, half + 1):
        x = frac - offset
        window = np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - (x / denominator) ** 2))) / np.i0(beta)
        # sinc(x) = sin(pi*x)/(pi*x), from the hoisted sine. `x == 0` happens only at `frac == 0`
        # and `offset == 0`, where sinc is 1; the divisor is substituted there rather than the
        # quotient patched afterwards, so the invalid division never happens at all.
        numerator = hoisted_sine if offset % 2 == 0 else -hoisted_sine
        at_zero = x == 0.0
        weight = np.where(at_zero, 1.0, numerator / (np.pi * np.where(at_zero, 1.0, x))) * window
        gathered = samples[np.clip(base_int + offset, 0, n - 1)]
        total += weight * gathered
        weight_sum += weight

    interpolated = np.where(weight_sum != 0.0, total / weight_sum, 0.0)
    # The position is inside [0, n - 1] when its whole part is, and the last sample is reached only exactly.
    in_range = (base_int >= 0) & ((base_int < n - 1) | ((base_int == n - 1) & (frac == 0.0)))
    return np.where(in_range, interpolated, 0.0)


#: Absolute accuracy the polynomial window must reach against the exact Kaiser window. Chosen
#: against what it protects rather than for roundness: the kernel's own error is 4.027e-12 of peak
#: (127 taps, beta = 32, measured against an analytic sinusoid), so 1e-13 leaves a factor of 40
#: between the approximation and the thing it must not disturb. It is also reachable -- float64
#: evaluation of a Chebyshev series bottoms out near 2e-14 at this beta, so the target sits above
#: the arithmetic's own floor rather than chasing it.
_WINDOW_FIT_TOLERANCE = 1e-13

#: Highest degree tried before giving up. A degree this large means the request is outside what a
#: polynomial can do in float64, and falling back is better than shipping a silently worse window.
_WINDOW_FIT_MAX_DEGREE = 64


def kaiser_window_chebyshev(beta: float, tolerance: float = _WINDOW_FIT_TOLERANCE) -> tuple[float, ...] | None:
    """Cache-normalising wrapper; see :func:`_kaiser_window_chebyshev` for the fit itself.

    ``float()`` before the cache, on a reviewer's finding: ``32`` and ``32.0`` are distinct keys to
    ``lru_cache`` but the same Kaiser window, so an int caller silently paid for a second fit and
    filled a slot. numpy scalars have the same problem.
    """
    return _kaiser_window_chebyshev(float(beta), float(tolerance))


@lru_cache(maxsize=32)
def _kaiser_window_chebyshev(beta: float, tolerance: float) -> tuple[float, ...] | None:
    """Return Chebyshev coefficients for the Kaiser window as a function of ``v = (x / denom) ** 2``.

    The device kernel evaluates the window ``i0(beta * sqrt(1 - v)) / i0(beta)`` once per tap, 127
    times per output sample, and that kernel is about 87% of the whole pipeline. Because ``i0`` is
    even, ``i0(sqrt(w))`` is analytic in ``w``: the window is a smooth function of ``v`` alone, so a
    polynomial in ``v`` replaces **both** the ``i0`` and the ``sqrt``. Measured on an RTX 3060 in
    float64, that is **2.18x** on the full kernel, with the output error against an analytic sinusoid
    unchanged at 4.027e-12 of peak and the two kernels differing by 2.2e-15.

    Fitted here rather than hard-coded because ``beta`` is a caller's choice, and a table for one
    beta would silently apply the wrong window to every other. ``beta`` is static at trace time, the fit is pure
    NumPy on the host, and the result is cached, so a run pays for it once per distinct beta.

    **Evaluated in the Chebyshev basis, never converted to monomials.** Converting a degree-24 fit to
    the monomial basis and using Horner cost three orders of accuracy when measured -- 1.9e-14
    against 1.8e-11 -- because that conversion is ill-conditioned. The recurrence costs roughly two
    fused multiply-adds per degree instead of one, which is why the measured speedup is 2.18x rather
    than the 3.66x a window-only comparison showed.

    Args:
        beta: Kaiser shape parameter the caller will use.
        tolerance: Largest acceptable absolute deviation from the exact window.

    **The bound is measured with a margin, not proved.** Acceptance samples an independent, denser
    grid than the fit used and requires half the tolerance, so a small exceedance between sample
    points cannot slip through as it once did. A minimax construction would give a genuine bound;
    this does not claim one. Two independent sweeps across the whole reachable region -- beta up to
    about 274, since ``validate_kernel`` caps beta at roughly ``(taps + 1) / 4`` and taps is unbounded
    -- found **no accepted fit over the target**, worst 0.68x at beta around 263. The default beta of
    32 measures 0.074x.

    **``None`` is reachable in ordinary use, and the boundary is not monotone.** At the default
    tolerance every beta below about 250 is served by a polynomial, but above that acceptance
    alternates -- 260 accepted at degree 64, 262 not, 264 accepted, 266 through 270 not, 272
    accepted -- because degree 64's error oscillates with beta rather than decreasing. A caller in
    that band gets the ``i0`` path and **no speedup**, and bisecting for a clean threshold will
    mislead, as it did during review. Such a configuration needs taps of roughly 1000 or more, so it
    is far from the default, but it is legal and it is not an error.

    Returns:
        Coefficients in ascending Chebyshev order for the domain ``[0, 1]``, or ``None`` when no
        degree up to :data:`_WINDOW_FIT_MAX_DEGREE` reaches half of ``tolerance`` on the independent
        grid -- in which case the caller must keep evaluating ``i0`` rather than accept a worse
        window.
    """
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError(f"Kaiser beta must be finite and non-negative; got {beta}.")

    def window(v: np.ndarray) -> np.ndarray:
        return np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - v))) / np.i0(beta)

    def sampling(uniform: int, decades: int, offset: float) -> np.ndarray:
        """Uniform points plus logarithmic clusters at **both** ends of [0, 1].

        Both ends, on a reviewer's counterexample. The first version clustered only at ``v -> 1``,
        where the window is smallest -- but the polynomial's difficulty is at ``v -> 0``, and at
        ``beta = 16.046`` an independent search found 1,243 points over target there, worst
        1.03e-13, while the fitting grid saw 9.97e-14 and accepted.
        """
        return np.unique(
            np.concatenate(
                [
                    np.linspace(0.0, 1.0, uniform),
                    np.logspace(-14.0 + offset, -1.0, decades),
                    1.0 - np.logspace(-14.0 + offset, -1.0, decades),
                ]
            ).clip(0.0, 1.0)
        )

    # Two grids, deliberately different points: fitting on one and accepting on the other is what
    # makes the check independent rather than circular. Accepting a fit against the very points it
    # minimised over is how the first version of this passed while exceeding its target between them.
    fit_grid = sampling(4001, 600, 0.0)
    check_grid = sampling(7919, 901, 0.37)  # coprime-ish counts and an offset, so the points differ
    fit_exact = window(fit_grid)
    check_exact = window(check_grid)

    # Half the target on the independent grid. A margin, because sampling can only ever bound the
    # error between its points: this is a measured bound with headroom, not a proof, and the
    # docstring says so rather than calling 1e-13 guaranteed.
    #
    # The margin's size is measured, and by adversarial sweeps rather than by this file's author --
    # who got it wrong twice. Worst inter-grid slippage found by an independent search: **1.54x**
    # (check grid against a denser one, near beta = 260-280 at degree 64), so a fit accepted at
    # 0.5 * tolerance can be up to ~0.77 * tolerance in truth: the 2x margin holds with about 1.3x
    # residual headroom. My own earlier notes put that slippage at 3% and then 1.37x; both came from
    # measuring only where my own sweep peaked.
    #
    # Why the margin is not sampling luck: `i0` is even, so the sqrt branch cancels and
    # `i0(beta * sqrt(1 - v)) / i0(beta)` is analytic on [0, 1]. Every extremum of the error is
    # therefore wider than the local grid spacing, and the peak near v -> 0 sits at beta * v ~ 0.6,
    # which the check grid's logarithmic clustering resolves with ~1e4 points even at beta = 274.
    acceptance = 0.5 * tolerance

    for degree in range(8, _WINDOW_FIT_MAX_DEGREE + 1, 2):
        fit = np.polynomial.chebyshev.Chebyshev.fit(fit_grid, fit_exact, degree, domain=[0.0, 1.0])
        if float(np.max(np.abs(fit(check_grid) - check_exact))) <= acceptance:
            return tuple(float(c) for c in fit.coef)
    return None
