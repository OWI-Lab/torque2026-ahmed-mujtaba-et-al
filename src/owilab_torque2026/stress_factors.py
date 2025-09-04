# -*- coding: utf-8 -*-
"""Stress factor calculations for OWI-Lab Torque 2026."""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

# Support importing both as part of the package and as a standalone module (e.g., doctest).
try:
    from . import geometry  # type: ignore[relative-beyond-top-level]
except ImportError:
    try:
        from owilab_torque2026 import geometry  # type: ignore[no-redef]
    except ImportError:
        import importlib
        geometry = importlib.import_module("geometry")


def stress_concentration_factor(
    outer_diameter: ArrayLike,
    t_small: ArrayLike,
    t_big: ArrayLike,
    cans_misalignment: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DNV RP-C203 Section 3.3.7.3-based simplified SCF formulation.

    Parameters
    ----------
    outer_diameter : ArrayLike
        Outer diameter D (broadcastable).
    t_small : ArrayLike
        Smaller thickness t (broadcastable).
    t_big : ArrayLike
        Larger thickness T (broadcastable).
    cans_misalignment : ArrayLike
        Cans misalignment (broadcastable).

    Returns
    -------
    tuple of float
        Tuple (scf_in, scf_out).

    Examples
    --------
    >>> scf_in, scf_out = stress_concentration_factor(
    ...     5000.0, 50.0, 60.0, 1.0
    ... )
    >>> isinstance(scf_in, float) and isinstance(scf_out, float)
    True
    """
    D = np.asarray(outer_diameter, dtype=float)
    t = np.asarray(t_small, dtype=float)
    T = np.asarray(t_big, dtype=float)
    delta_m = np.asarray(cans_misalignment, dtype=float)

    delta_nt = 0.05 * t
    delta_t = (T - t) / 2.0
    transition_length = (T - t) * 4.0

    log_ratio = np.log10(D / t)
    beta = 1.5 - (1.0 / log_ratio) + 3.0 / (log_ratio**2)
    alpha = (
        1.82
        * transition_length
        / (np.sqrt(D * t) * (1.0 + (T / t) ** beta))
    )

    exp_term = np.exp(-alpha) / (1.0 + (T / t) ** beta)
    scf_in = 1.0 + 6.0 * ((delta_t + delta_m - delta_nt) / t) * exp_term
    scf_out = 1.0 - 6.0 * ((delta_t - delta_m + delta_nt) / t) * exp_term
    return scf_in, scf_out


def scale_effect(
    t: ArrayLike,
    t_ref: float,
    t_eff_allowance: float,
    t_corr_exponent: float,
    weld_width: ArrayLike,
) -> np.ndarray:
    """
    DNV-RP-C-203 effective thickness based scale effect.

    t_eff = min(14 + 0.66*(weld_width + t_eff_allowance), t)
    scale_factor = (t_eff/t_ref)**t_corr_exponent

    Parameters
    ----------
    t : ArrayLike
        Plate thickness.
    t_ref : float
        Reference plate thickness.
    t_eff_allowance : float
        Effective thickness allowance.
    t_corr_exponent : float
        Thickness correction exponent.
    weld_width : ArrayLike
        Weld width.

    Returns
    -------
    numpy.ndarray
        Scale factor.

    Examples
    --------
    >>> float(scale_effect(20.0, 40.0, 2.0, 2.0, 10.0))
    0.25
    """
    t = np.asarray(t, dtype=float)
    weld_width = np.asarray(weld_width, dtype=float)

    t_eff = np.minimum(14.0 + 0.66 * (weld_width + t_eff_allowance), t)
    scale_factor = (t_eff / t_ref) ** t_corr_exponent
    return scale_factor


def smf_inner_outer(
    outer_diameter: ArrayLike,
    t_small: ArrayLike,
    t_big: ArrayLike,
    cans_misalignment: ArrayLike,
    t_ref: float,
    t_eff_allowance: float,
    t_corr_exponent: float,
    weld_width: ArrayLike,
    material_factor: float = 1.25,
    section_modulus_reference: Literal["inner", "outer"] = "inner",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Stress Multiplication Factors (SMF) for inner and outer.

    The section modulus reference defines which section modulus is used to
    compute structural extrapolation factors (SEF).

    Parameters
    ----------
    outer_diameter : ArrayLike
        Outer diameter D.
    t_small : ArrayLike
        Smaller thickness t.
    t_big : ArrayLike
        Larger thickness T.
    cans_misalignment : ArrayLike
        Cans misalignment.
    t_ref : float
        Reference thickness for scale effect.
    t_eff_allowance : float
        Allowance for effective thickness.
    t_corr_exponent : float
        Corrosion exponent for thickness correction.
    weld_width : ArrayLike
        Weld width.
    material_factor : float, default=1.25
        Material factor multiplier.
    section_modulus_reference : {'inner', 'outer'}, default='inner'
        Reference for SEF computation.

    Returns
    -------
    tuple of numpy.ndarray
        Tuple (smf_in, smf_out).

    Examples
    --------
    >>> smf_in, smf_out = smf_inner_outer(
    ...     5000.0, 50.0, 60.0, 0.0, 40.0, 2.0, 2.0, 32.0
    ... )
    >>> float(smf_in) > 0 and float(smf_out) > 0
    True
    """
    D = np.asarray(outer_diameter, dtype=float)
    t = np.asarray(t_small, dtype=float)

    scf_in, scf_out = stress_concentration_factor(
        D, t, t_big, cans_misalignment
    )
    sc = scale_effect(t, t_ref, t_eff_allowance, t_corr_exponent, weld_width)

    z_out = geometry.section_modulus_outer(D, t)
    z_in = geometry.section_modulus_inner(D, t)

    z_ref = z_in if section_modulus_reference == "inner" else z_out

    sef_in = z_ref / z_in
    sef_out = z_ref / z_out

    smf_in = scf_in * sc * sef_in * material_factor
    smf_out = scf_out * sc * sef_out * material_factor
    return smf_in, smf_out


def build_smf_dataframe(
    t_nom: float,
    t_tol_up: float,
    t_big_nom: float,
    t_big_tol_up: float,
    d_nom: float,
    d_tol_low: float,
    d_tol_up: float,
    l_cm: float,
    u_cm: float,
    t_ref: float,
    t_eff_allowance: float,
    t_corr_exponent: float,
    material_factor: float = 1.25,
    n_t: int = 20,
    n_t_big: int = 20,
    n_d: int = 21,
    n_mis: int = 21,
) -> pd.DataFrame:
    """
    Build grids of dimensions, compute SMFs and return a flattened DataFrame.

    The output DataFrame column names are kept for compatibility.

    Parameters
    ----------
    t_nom : float
        Nominal smaller thickness.
    t_tol_up : float
        Upper tolerance for smaller thickness.
    t_big_nom : float
        Nominal larger thickness.
    t_big_tol_up : float
        Upper tolerance for larger thickness.
    d_nom : float
        Nominal outer diameter.
    d_tol_low : float
        Lower tolerance for outer diameter.
    d_tol_up : float
        Upper tolerance for outer diameter.
    l_cm : float
        Lower bound for misalignment.
    u_cm : float
        Upper bound for misalignment.
    t_ref : float
        Reference thickness for scale effect.
    t_eff_allowance : float
        Allowance for effective thickness.
    t_corr_exponent : float
        Corrosion exponent for thickness correction.
    material_factor : float, default=1.25
        Material factor multiplier.
    n_t : int, default=20
        Number of points in smaller thickness grid.
    n_t_big : int, default=20
        Number of points in larger thickness grid.
    n_d : int, default=21
        Number of points in diameter grid.
    n_mis : int, default=21
        Number of points in misalignment grid.

    Returns
    -------
    pandas.DataFrame
        Flattened DataFrame with geometry, factors and SMFs.

    Notes
    -----
    Weld width is set to 0.64 * t.

    Examples
    --------
    >>> df_test = build_smf_dataframe(
    ...     50.0, 2.0, 60.0, 2.0, 5000.0, -10.0, 10.0,
    ...     -5.0, 5.0, 40.0, 2.0, 2.0,
    ...     n_t=2, n_t_big=2, n_d=3, n_mis=3
    ... )
    >>> len(df_test)
    36
    >>> {'t', 'T', 'D', 'misalignment'}.issubset(df_test.columns)
    True
    >>> (df_test['SMF_IN'] > 0).all() and (df_test['SMF_OUT'] > 0).all()
    np.True_
    """
    t_range = np.linspace(t_nom, t_nom + t_tol_up, n_t)
    t_big_range = np.linspace(t_big_nom, t_big_nom + t_big_tol_up, n_t_big)
    d_range = np.linspace(d_nom + d_tol_low, d_nom + d_tol_up, n_d)
    misalignment_range = np.linspace(l_cm, u_cm, n_mis)

    t_grid, t_big_grid, d_grid, mis_grid = np.meshgrid(
        t_range, t_big_range, d_range, misalignment_range, indexing="ij"
    )

    weld_width_grid = 0.64 * t_grid
    i_grid = geometry.moment_of_inertia(d_grid, t_grid)
    scf_in_grid, scf_out_grid = stress_concentration_factor(
        d_grid, t_grid, t_big_grid, mis_grid
    )
    scale_f_grid = scale_effect(
        t_grid, t_ref, t_eff_allowance, t_corr_exponent, weld_width_grid
    )

    section_modulus_out_grid = i_grid / (d_grid / 2)
    section_modulus_in_grid = i_grid / ((d_grid - 2 * t_grid) / 2)

    i_nominal = geometry.moment_of_inertia(d_nom, t_nom)
    section_modulus_nom_in = i_nominal / ((d_nom - 2 * t_nom) / 2)

    sef_out_grid = section_modulus_nom_in / section_modulus_out_grid
    sef_in_grid = section_modulus_nom_in / section_modulus_in_grid

    smf_in_grid = scf_in_grid * scale_f_grid * sef_in_grid * material_factor
    smf_out_grid = scf_out_grid * scale_f_grid * sef_out_grid * material_factor

    df_out = pd.DataFrame(
        {
            # Keep original column names for compatibility
            "t": t_grid.ravel(),
            "T": t_big_grid.ravel(),
            "D": d_grid.ravel(),
            "misalignment": mis_grid.ravel(),
            "I": i_grid.ravel(),
            "SCF_in": scf_in_grid.ravel(),
            "SCF_out": scf_out_grid.ravel(),
            "scale_f": scale_f_grid.ravel(),
            "section_mod_out": section_modulus_out_grid.ravel(),
            "section_mod_in": section_modulus_in_grid.ravel(),
            "SEF_in": sef_in_grid.ravel(),
            "SEF_out": sef_out_grid.ravel(),
            "SMF_IN": smf_in_grid.ravel(),
            "SMF_OUT": smf_out_grid.ravel(),
        }
    )
    return df_out
