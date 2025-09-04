# -*- coding: utf-8 -*-
"""Fatigue analysis functions for OWI-Lab Torque 2026.

This module provides utilities for deterministic fatigue damage and life
estimation using a two-slope S-N curve and Weibull stress range models.

Examples
--------
Compute fatigue life from given damage at default FDF 3.0.

>>> round(fatigue_life_from_damage(0.25), 3)
np.float64(1.333)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt
import pandas as pd
from py_fatigue import SNCurve
from scipy.special import gamma, gammaincc  # pylint: disable=E0611
from scipy.stats import weibull_min

if TYPE_CHECKING:
    from _typeshed import ConvertibleToFloat

def fatigue_damage_two_slope(
    q_eff: npt.ArrayLike,
    s_eff: npt.ArrayLike,
    m1: npt.ArrayLike,
    m2: npt.ArrayLike,
    a1: npt.ArrayLike,
    a2: npt.ArrayLike,
    h: npt.ArrayLike,
    n: npt.ArrayLike,
) -> np.ndarray | float:
    """
    Deterministic damage under Weibull stress range distribution and two-slope
    S-N curve.

    Parameters
    ----------
    q_eff : ArrayLike
        Effective Weibull scale (possibly stress-mean corrected).
    s_eff : ArrayLike
        Effective knee stress (possibly stress-mean corrected).
    m1 : ArrayLike
        S-N slope below the knee (positive).
    m2 : ArrayLike
        S-N slope above the knee (positive).
    a1 : ArrayLike
        S-N intercept below the knee (same units as stress^m1).
    a2 : ArrayLike
        S-N intercept above the knee (same units as stress^m2).
    h : ArrayLike
        Weibull shape parameter (k).
    n : ArrayLike
        Number of stress cycles.

    Returns
    -------
    ndarray or float
        Fatigue damage (Miner's sum). Scalar if inputs are broadcastable to a
        scalar, otherwise an array per NumPy broadcasting rules.

    Notes
    -----
    The formula integrates the S-N relationship over a Weibull stress range
    distribution. It splits the integral at the knee stress and uses the
    regularized upper incomplete gamma function.

    Examples
    --------
    For zero knee stress, the upper branch contribution is zero.

    >>> round(float(fatigue_damage_two_slope(2.0, 0.0, 3.0, 5.0,
    ...                                      1.0, 1.0, 2.0, 1.0)), 6)
    10.634723

    Broadcasting over the number of cycles:

    >>> fatigue_damage_two_slope(2.0, 0.0, 3.0, 5.0, 2.0, 4.0, 2.0,
    ...                          [1.0, 2.0]).shape
    (2,)

    Damage increases with increasing scale parameter:

    >>> d1 = fatigue_damage_two_slope(1.0, 0.0, 3.0, 5.0, 1.0, 1.0, 2.0, 1.0)
    >>> d2 = fatigue_damage_two_slope(2.0, 0.0, 3.0, 5.0, 1.0, 1.0, 2.0, 1.0)
    >>> d2 > d1
    True
    """
    # Convert inputs to arrays to support broadcasting and avoid list * float
    # errors.
    q_eff = np.asarray(q_eff, dtype=float)
    s_eff = np.asarray(s_eff, dtype=float)
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    h = np.asarray(h, dtype=float)
    n = np.asarray(n, dtype=float)

    g1 = gamma(1 + m1 / h) * gammaincc(1 + m1 / h, (s_eff / q_eff) ** h)
    g2 = gamma(1 + m2 / h) * (1.0 - gammaincc(1 + m2 / h, (s_eff / q_eff) ** h))
    d = n * ((q_eff**m1) / a1 * g1 + (q_eff**m2) / a2 * g2)

    # Return a Python float for scalar outputs.
    if np.ndim(d) == 0:
        return float(d)
    return d


def fatigue_life_from_damage(
    damage: npt.ArrayLike, fdf: float = 3.0
) -> ConvertibleToFloat:
    """
    Convert damage into fatigue life (number of cycles) using an FDF.

    Parameters
    ----------
    damage : ArrayLike
        Computed fatigue damage (Miner's sum).
    fdf : float, default=3.0
        Fatigue design factor. Life is scaled by 1 / fdf.

    Returns
    -------
    ndarray or float
        Number of cycles to failure corresponding to the given damage and FDF.

    Examples
    --------
    Scalar example:

    >>> round(fatigue_life_from_damage(0.2), 3)
    np.float64(1.667)

    Vectorized input:

    >>> fatigue_life_from_damage(np.array([0.5, 0.25]), fdf=2.0).tolist()
    [1.0, 2.0]
    """
    damage = np.asarray(damage, dtype=float)
    return (1.0 / fdf) * (1.0 / damage)


def solve_weibull_scale_for_damage(
    m1: float,
    m2: float,
    log_a1: float,
    log_a2: float,
    s_knee: ConvertibleToFloat,
    h: float,
    t_years: float,
    n_per_year: float,
    fdf: float,
    smf: ConvertibleToFloat,
    q0: float = 2.0,
    tol: float = 1e-6,
    step: float = 1e-4,
    max_iter: int = 2_000_000,
) -> ConvertibleToFloat:
    """
    Solve for Weibull scale parameter q such that damage over ``t_years``
    equals ``1 / fdf``. Both knee stress and Weibull scale are scaled by
    ``smf``.

    Parameters
    ----------
    m1 : float
        S-N slope below the knee.
    m2 : float
        S-N slope above the knee.
    log_a1 : float
        Base-10 logarithm of the S-N intercept below the knee.
    log_a2 : float
        Base-10 logarithm of the S-N intercept above the knee.
    s_knee : float or ArrayLike
        Knee stress.
    h : float
        Weibull shape parameter (k).
    t_years : float
        Duration in years over which damage is accumulated.
    n_per_year : float
        Number of cycles per year.
    fdf : float
        Fatigue design factor.
    smf : float or ArrayLike
        Stress mean factor scaling applied to both ``q`` and ``s_knee``.
    q0 : float, default=2.0
        Initial guess for the Weibull scale. It should be below the solution
        because the algorithm only increments ``q``.
    tol : float, default=1e-6
        Stopping tolerance on the difference ``(1/fdf - damage)``.
    step : float, default=1e-4
        Increment step for ``q`` per iteration.
    max_iter : int, default=2_000_000
        Maximum number of iterations.

    Returns
    -------
    ArrayLike
        The Weibull scale parameter ``q`` meeting the target damage.

    Notes
    -----
    This is a simple forward-increment solver. If ``q0`` is too large,
    the loop exits immediately and returns ``q0`` without refinement.

    Examples
    --------
    With ``s_knee = 0``, the target damage is solved quickly. Here, the
    analytical target is approximately ``q ≈ 0.9096``.

    >>> q = solve_weibull_scale_for_damage(
    ...     m1=3.0, m2=5.0, log_a1=0.0, log_a2=0.0, s_knee=0.0, h=2.0,
    ...     t_years=1.0, n_per_year=1.0, fdf=1.0, smf=1.0, q0=0.0,
    ...     step=1e-3, tol=1e-6, max_iter=100000,
    ... )
    >>> round(q, 3)
    0.911

    The resulting damage is within a small tolerance of the target:

    >>> d = fatigue_damage_two_slope(q, 0.0, 3.0, 5.0, 1.0, 1.0, 2.0, 1.0)
    >>> np.isclose(d, 1.0, atol=6e-3)
    np.True_
    """
    a1, a2 = 10**log_a1, 10**log_a2
    d_crit = 1.0 / fdf

    q = float(q0)
    q = np.asarray(q, dtype=float)
    s_knee = np.asarray(s_knee, dtype=float)
    thr = 1.0
    it = 0
    while thr > tol and it < max_iter:
        q_eff = q * smf
        s_eff = s_knee * smf

        if q_eff <= 0.0:
            d = 0.0
        else:
            # x = (s_eff / q_eff) ** h
            d = fatigue_damage_two_slope(q_eff=q_eff, s_eff=s_eff, m1=m1, m2=m2,
                                         a1=a1, a2=a2, h=h,
                                         n=t_years * n_per_year)

        thr = d_crit - d
        q += step
        it += 1
    # Return a Python float for scalar outputs.
    if np.ndim(q) == 0:
        return float(q)
    return q


def weibull_bin_pdf(
    bin_edges: npt.ArrayLike, k_shape: float, scale: ConvertibleToFloat
) -> np.ndarray:
    """
    Discrete PDF for bins from differences of the Weibull CDF.

    Sum over bins equals ``CDF(last_edge) - CDF(first_edge)``. Including
    ``np.inf`` as the last edge makes the sum equal to 1.

    Parameters
    ----------
    bin_edges : ArrayLike
        Monotonic array of bin edges (length >= 2).
    k_shape : float
        Weibull shape parameter (k).
    scale : float
        Weibull scale parameter (λ).

    Returns
    -------
    ndarray
        Probability per bin (length ``len(bin_edges) - 1``).

    Examples
    --------
    Using an infinite upper edge makes the probabilities sum to one:

    >>> pdf = weibull_bin_pdf([0.0, 1.0, np.inf], k_shape=2.0, scale=1.0)
    >>> pdf.shape
    (2,)
    >>> float(pdf.sum())
    1.0
    >>> np.all(pdf >= 0)
    np.True_

    With finite edges, the sum is less than 1:

    >>> weibull_bin_pdf([0.0, 1.0, 2.0], 2.0, 1.0).sum() < 1.0
    np.True_
    """
    x = np.asarray(bin_edges, dtype=float)
    cdf = weibull_min.cdf(x, c=k_shape, scale=scale)
    pdf = np.diff(cdf)
    return pdf

def compute_fatigue_life(
    df: pd.DataFrame,
    q_mean: float,
    sn: SNCurve,
    h: float,
    n_cycles: float,
    fdf: float,
    s_knee: float,
) -> pd.DataFrame:
    """
    Compute fatigue life for inside and outside surfaces.

    The function adds two columns to df:
    - 'Fatigue_life_IN'
    - 'Fatigue_life_OUT'

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing SMFs. Must have 'SMF_IN' and 'SMF_OUT'.
    q_mean : float
        Weibull scale for the stress range at the reference location.
    sn : SNCurve
        SN curve object providing slopes and intercepts.
    h : float
        Weibull shape parameter for stress range distribution.
    n_cycles : float
        Number of stress cycles over the design life.
    fdf : float
        Fatigue design factor.
    s_knee : float
        Knee stress from the SN curve.

    Returns
    -------
    pandas.DataFrame
        The same dataframe with added fatigue life columns.

    Notes
    -----
    This function modifies the input dataframe in place.

    Examples
    --------
    >>> import pandas as pd
    >>> test_df = pd.DataFrame(
    ...     {'SMF_IN': [1.0, 1.5], 'SMF_OUT': [1.2, 0.8]}
    ... )
    >>> class DummySN:
    ...     slope = [3.0, 5.0]
    ...     intercept = [11.0, 12.0]
    ...     def get_knee_stress(self):
    ...         return [80.0]
    ...
    >>> out = compute_fatigue_life(
    ...     test_df.copy(), q_mean=4.0, sn=DummySN(), h=0.8,
    ...     n_cycles=1e6, fdf=3.0, s_knee=80.0
    ... )
    >>> set(['Fatigue_life_IN', 'Fatigue_life_OUT']).issubset(out.columns)
    True
    >>> (out[['Fatigue_life_IN', 'Fatigue_life_OUT']] > 0).all().all()
    np.True_
    """
    m1, m2 = sn.slope
    loga1, loga2 = sn.intercept
    a1, a2 = 10.0 ** loga1, 10.0 ** loga2

    # Inside
    smf_in = df['SMF_IN'].to_numpy(dtype=float)
    q_in = q_mean * smf_in
    s_in = s_knee * smf_in
    damage_in = fatigue_damage_two_slope(
        q_eff=q_in, s_eff=s_in, m1=m1, m2=m2, a1=a1, a2=a2, h=h, n=n_cycles
    )
    df['Fatigue_life_IN'] = fatigue_life_from_damage(damage_in, fdf=fdf)

    # Outside
    smf_out = df['SMF_OUT'].to_numpy(dtype=float)
    q_out = q_mean * smf_out
    s_out = s_knee * smf_out
    damage_out = fatigue_damage_two_slope(
        q_eff=q_out, s_eff=s_out, m1=m1, m2=m2, a1=a1, a2=a2, h=h, n=n_cycles
    )
    df['Fatigue_life_OUT'] = fatigue_life_from_damage(damage_out, fdf=fdf)

    return df
