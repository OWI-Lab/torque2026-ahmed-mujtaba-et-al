# -*- coding: utf-8 -*-
"""Geometric properties for OWI-Lab Torque 2026.
This module provides utilities for calculating geometric properties of hollow
circular sections, including second moment of area and section modulus at inner
and outer fibers.

Examples
--------
Compute second moment of area for a hollow circular section with outer diameter
10 units and wall thickness 1 unit.

>>> import numpy as np
>>> np.isclose(
...     moment_of_inertia(10.0, 1.0),
...     (np.pi / 64.0) * (10.0**4 - 8.0**4),
... )
np.True_
>>> D = np.array([10.0, 12.0])
>>> t = 1.0
>>> expected = (np.pi / 64.0) * (D**4 - (D - 2.0 * t) ** 4)
>>> np.allclose(moment_of_inertia(D, t), expected)
True
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_array(x: ArrayLike) -> NDArray[np.float64]:
    """Return input as a float64 numpy array.

    Parameters
    ----------
    x : ArrayLike
        Input value(s).

    Returns
    -------
    numpy.ndarray
        Array of dtype float64.

    Examples
    --------
    >>> _as_array(3).dtype == np.float64
    True
    >>> _as_array([1, 2, 3]).shape
    (3,)
    """
    return np.asarray(x, dtype=float)


def moment_of_inertia(
    outer_diameter: ArrayLike, wall_thickness: ArrayLike
) -> NDArray[np.float64]:
    """Second moment of area for a hollow circular section.

    Computed about the neutral axis.

    Parameters
    ----------
    outer_diameter : ArrayLike
        Outer diameter (same units).
    wall_thickness : ArrayLike
        Wall thickness (same units).

    Returns
    -------
    numpy.ndarray
        Second moment of area in units^4.

    Examples
    --------
    >>> import numpy as np
    >>> np.isclose(
    ...     moment_of_inertia(10.0, 1.0),
    ...     (np.pi / 64.0) * (10.0**4 - 8.0**4),
    ... )
    np.True_
    >>> D = np.array([10.0, 12.0])
    >>> t = 1.0
    >>> expected = (np.pi / 64.0) * (D**4 - (D - 2.0 * t) ** 4)
    >>> np.allclose(moment_of_inertia(D, t), expected)
    True
    """
    outer_diameter_arr = _as_array(outer_diameter)
    wall_thickness_arr = _as_array(wall_thickness)
    inner_diameter = outer_diameter_arr - 2.0 * wall_thickness_arr
    second_moment = (np.pi / 64.0) * (
        outer_diameter_arr**4 - inner_diameter**4
    )
    return second_moment


def section_modulus_outer(
    outer_diameter: ArrayLike, wall_thickness: ArrayLike
) -> NDArray[np.float64]:
    """Section modulus at the outside fiber.

    Computed as I / (D / 2).

    Parameters
    ----------
    outer_diameter : ArrayLike
        Outer diameter (same units).
    wall_thickness : ArrayLike
        Wall thickness (same units).

    Returns
    -------
    numpy.ndarray
        Section modulus in units^3.

    Examples
    --------
    >>> import numpy as np
    >>> D = 10.0
    >>> t = 1.0
    >>> np.isclose(
    ...     section_modulus_outer(D, t),
    ...     moment_of_inertia(D, t) / (D / 2.0),
    ... )
    np.True_
    >>> section_modulus_outer([10.0, 12.0], 1.0).shape
    (2,)
    """
    outer_diameter_arr = _as_array(outer_diameter)
    second_moment = moment_of_inertia(outer_diameter_arr, wall_thickness)
    return second_moment / (outer_diameter_arr / 2.0)


def section_modulus_inner(
    outer_diameter: ArrayLike, wall_thickness: ArrayLike
) -> NDArray[np.float64]:
    """Section modulus at the inside fiber.

    Computed as I / ((D - 2t) / 2).

    Parameters
    ----------
    outer_diameter : ArrayLike
        Outer diameter (same units).
    wall_thickness : ArrayLike
        Wall thickness (same units).

    Returns
    -------
    numpy.ndarray
        Section modulus in units^3.

    Examples
    --------
    >>> import numpy as np
    >>> D = 10.0
    >>> t = 1.0
    >>> np.isclose(
    ...     section_modulus_inner(D, t),
    ...     moment_of_inertia(D, t) / ((D - 2.0 * t) / 2.0),
    ... )
    np.True_
    >>> section_modulus_inner([10.0, 12.0], 1.0).shape
    (2,)
    """
    outer_diameter_arr = _as_array(outer_diameter)
    wall_thickness_arr = _as_array(wall_thickness)
    second_moment = moment_of_inertia(outer_diameter_arr, wall_thickness_arr)
    return second_moment / (
        (outer_diameter_arr - 2.0 * wall_thickness_arr) / 2.0
    )
