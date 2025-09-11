# -*- coding: utf-8 -*-
"""Plotting utilities for OWI-Lab Torque 2026."""
from __future__ import annotations

import logging
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class PdfLike(Protocol):  # pylint: disable=too-few-public-methods
    """Protocol for objects exposing a .pdf method."""

    def pdf(self, x: np.ndarray) -> np.ndarray:  
        """Method returning the PDF evaluated at x."""
        return np.array([])  # pragma: no cover


def truncated_norms_t1_t2_d(
    trunc_t1: PdfLike,
    trunc_t2: PdfLike,
    trunc_d: PdfLike,
    l_t1: float,
    u_t1: float,
    l_t2: float,
    u_t2: float,
    mean_target_d: float,
    d_tol_low: float,
    d_tol_up: float,
    *args: Any,
    **kwargs: Any,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """
    Plot PDFs for t1, t2 and d truncated normals on dual x-axes.

    The two thickness PDFs (t1 and t2) are drawn on the bottom x-axis,
    while the diameter PDF (d) is drawn on a twin top x-axis.

    Parameters
    ----------
    trunc_t1
        Distribution-like object with a pdf method for t1.
    trunc_t2
        Distribution-like object with a pdf method for t2.
    trunc_d
        Distribution-like object with a pdf method for d.
    l_t1
        Lower bound for t1.
    u_t1
        Upper bound for t1.
    l_t2
        Lower bound for t2.
    u_t2
        Upper bound for t2.
    mean_target_d
        Target mean diameter used to define the d sampling window.
    d_tol_low
        Lower deviation from mean_target_d for d sampling window.
    d_tol_up
        Upper deviation from mean_target_d for d sampling window.
    *args
        Forwarded to matplotlib plot.
    **kwargs
        Forwarded to matplotlib plot. Recognized keys:
        - fig: optional matplotlib Figure to plot in.
        - figsize: figure size if fig is not provided.

    Returns
    -------
    (fig, (ax_bottom, ax_top))
        The created figure and a tuple with the two axes.

    Examples
    --------
    >>> import numpy as np
    >>> class Dummy:
    ...     def pdf(self, x):
    ...         return np.ones_like(x, dtype=float)
    >>> fig, (ax1, ax2) = truncated_norms_t1_t2_d(
    ...     Dummy(), Dummy(), Dummy(), 0.0, 1.0, 0.0, 2.0, 10.0, -1.0, 1.0
    ... )
    >>> isinstance(fig, plt.Figure)
    True
    >>> len(ax1.lines), len(ax2.lines)
    (2, 1)
    """
    fig = kwargs.pop("fig", None)
    if fig is None:
        fig = plt.figure(figsize=kwargs.pop("figsize", (16 / 2.54, 7 / 2.54)))
    ax1 = fig.add_subplot(111)

    x_t1 = np.linspace(l_t1, u_t1, 500)
    x_t2 = np.linspace(l_t2, u_t2, 500)
    ax1.plot(
        x_t1,
        trunc_t1.pdf(x_t1),
        *args,
        **{**{"lw": 2, "label": "t-PDF", "color": "b"}, **kwargs},
    )
    ax1.plot(
        x_t2,
        trunc_t2.pdf(x_t2),
        *args,
        **{**{"lw": 2, "label": "T-PDF", "color": "r"}, **kwargs},
    )
    ax1.set_xlabel("Thickness (mm)")
    ax1.set_ylabel("Probability Density")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twiny()
    x_d = np.linspace(mean_target_d + d_tol_low, mean_target_d + d_tol_up, 1000)
    ax2.plot(
        x_d,
        trunc_d.pdf(x_d),
        *args,
        **{**{"lw": 2, "label": "D-PDF", "color": "g"}, **kwargs},
    )
    ax2.set_xlabel("Diameter (mm)")
    ax2.legend(loc="upper right")
    return fig, (ax1, ax2)


def misalignment_pdf(
    samples_misalignment: np.ndarray | list[float],
    trunc_misalignment: PdfLike,
    l_cm: float,
    u_cm: float,
    *args: Any,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Plot histogram and PDF for misalignment truncated normal.

    Parameters
    ----------
    samples_misalignment
        Sample values of misalignment.
    trunc_misalignment
        Distribution-like object with a pdf method for misalignment.
    l_cm
        Lower bound of misalignment range.
    u_cm
        Upper bound of misalignment range.
    *args
        Forwarded to matplotlib plot.
    **kwargs
        Forwarded to matplotlib plot. Recognized keys:
        - bins: histogram bins (default 50)
        - fig: optional matplotlib Figure to plot in.
        - figsize: figure size if fig is not provided.

    Returns
    -------
    (fig, ax)
        The created figure and axes.

    Examples
    --------
    >>> import numpy as np
    >>> class Dummy:
    ...     def pdf(self, x):
    ...         return np.ones_like(x, dtype=float)
    >>> rng = np.random.default_rng(0)
    >>> samples = rng.normal(size=100)
    >>> fig, ax = misalignment_pdf(samples, Dummy(), -3.0, 3.0)
    >>> isinstance(fig, plt.Figure) and hasattr(ax, "plot")
    True
    """
    fig = kwargs.pop("fig", None)
    if fig is None:
        fig = plt.figure(figsize=kwargs.pop("figsize", (16 / 2.54, 7 / 2.54)))
    ax = fig.add_subplot(111)

    ax.hist(
        samples_misalignment,
        bins=kwargs.pop("bins", 50),
        density=True,
        alpha=0.5,
        label="Samples",
    )
    x_cm = np.linspace(l_cm, u_cm, 500)
    ax.plot(x_cm, trunc_misalignment.pdf(x_cm), "r-", lw=2,
            label="Truncated Normal PDF")
    ax.set_xlabel("Cans Misalignment (mm)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Truncated Normal Distribution for Cans Misalignment")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    return fig, ax


def weibull_pdf(
    bin_edges: np.ndarray,
    pdf_values: np.ndarray,
    *args: Any,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Plot a discrete Weibull PDF defined on bin edges (left edges for x).

    Parameters
    ----------
    bin_edges
        Array of bin edges. Left edges define the x locations.
    pdf_values
        PDF values aligned with left edges (len == len(bin_edges) - 1).
    *args
        Forwarded to matplotlib plot.
    **kwargs
        Forwarded to matplotlib plot. Recognized keys:
        - title: optional title.
        - xlim: x limits (default (0, 30)).
        - label: legend label.
        - fig: optional matplotlib Figure to plot in.
        - figsize: figure size if fig is not provided.

    Returns
    -------
    (fig, ax)
        The created figure and axes.

    Examples
    --------
    >>> import numpy as np
    >>> edges = np.array([0, 1, 2, 3], float)
    >>> pdf = np.array([0.2, 0.5, 0.3], float)
    >>> fig, ax = weibull_pdf(edges, pdf, title="weibull")
    >>> ax.get_xlim()[0] == 0
    np.True_
    """
    title = kwargs.pop("title", None)
    xlim = kwargs.pop("xlim", (0, 30))
    label = kwargs.pop("label", "Weibull PDF")

    fig = kwargs.pop("fig", None)
    if fig is None:
        fig = plt.figure(figsize=kwargs.pop("figsize", (16 / 2.54, 7 / 2.54)))
    ax = fig.add_subplot(111)

    ax.plot(
        bin_edges[:-1],
        pdf_values,
        *args,
        **{**{"linestyle": "--", "color": "black", "label": label}, **kwargs},
    )
    ax.set_xlim(xlim)
    ax.set_xlabel("Stress Range (MPa)")
    ax.set_ylabel("Probability Density (PDF)")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return fig, ax


def contour_life_t1_t2(
    pivot_df: pd.DataFrame,
    trunc_t1: PdfLike,
    trunc_t2: PdfLike,
    *args: Any,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Contour of fatigue life vs (t1, t2) with joint PDF background.

    Parameters
    ----------
    pivot_df
        DataFrame with index=t1 grid, columns=t2 grid, values=life.
    trunc_t1
        Distribution-like object with a pdf method for t1.
    trunc_t2
        Distribution-like object with a pdf method for t2.
    *args
        Unused. Reserved for future compatibility.
    **kwargs
        Plot customization. Recognized keys:
        - levels_pdf: int levels for PDF filled contour (default 30)
        - levels_contour: int levels for life contour (default 20)
        - cmap: colormap for PDF (default 'Blues')
        - title: optional title
        - inline_fontsize: fontsize for contour labels (default 8)
        - fig: optional matplotlib Figure
        - figsize: figure size if fig is not provided

    Returns
    -------
    (fig, ax)
        The created figure and axes.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> class Dummy:
    ...     def pdf(self, x):
    ...         return np.ones_like(x, dtype=float)
    >>> df = pd.DataFrame([[1., 2.], [3., 4.]],
    ...                   index=[1., 2.], columns=[10., 12.])
    >>> fig, ax = contour_life_t1_t2(
    ...     df, Dummy(), Dummy(), levels_contour=3
    ... )
    >>> isinstance(fig, plt.Figure)
    True
    """
    levels_pdf = kwargs.pop("levels_pdf", 30)
    levels_contour = kwargs.pop("levels_contour", 20)
    cmap = kwargs.pop("cmap", "Blues")
    title = kwargs.pop("title", None)
    inline_fontsize = kwargs.pop("inline_fontsize", 8)

    fig = kwargs.pop("fig", None)
    if fig is None:
        fig = plt.figure(figsize=kwargs.pop("figsize", (16 / 2.54, 7 / 2.54)))
    if "ax" in kwargs:
        ax = kwargs.pop("ax")
    else:
        ax = fig.add_subplot(111)

    t1_unique = pivot_df.index.values
    t2_unique = pivot_df.columns.values
    life_grid = pivot_df.values
    t1_mesh, t2_mesh = np.meshgrid(t1_unique, t2_unique, indexing="ij")

    t1_fine = np.linspace(t1_unique.min(), t1_unique.max(), 200)
    t2_fine = np.linspace(t2_unique.min(), t2_unique.max(), 200)
    t1_fine_mesh, t2_fine_mesh = np.meshgrid(t1_fine, t2_fine, indexing="ij")
    pdf_t1 = trunc_t1.pdf(t1_fine_mesh)
    pdf_t2 = trunc_t2.pdf(t2_fine_mesh)
    joint_pdf = pdf_t1 * pdf_t2

    _ = ax.contourf(
        t1_fine_mesh, t2_fine_mesh, joint_pdf, levels=levels_pdf, cmap=cmap,
        alpha=0.6,
    )
    cs = ax.contour(t1_mesh, t2_mesh, life_grid, levels=levels_contour, colors="k")
    ax.clabel(cs, inline=True, fontsize=inline_fontsize, fmt="%1.1f")
    ax.set_xlabel("Thickness (t) mm")
    ax.set_ylabel("Thickness (T) mm")
    if title:
        ax.set_title(title)
    return fig, ax


def contour_life_d_mis(
    pivot_df: pd.DataFrame,
    trunc_d: PdfLike,
    trunc_misalignment: PdfLike,
    *args: Any,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Contour of fatigue life vs (d, misalignment) with joint PDF background.

    Parameters
    ----------
    pivot_df
        DataFrame with index=d grid, columns=misalignment grid, values=life.
    trunc_d
        Distribution-like object with a pdf method for d.
    trunc_misalignment
        Distribution-like object with a pdf method for misalignment.
    *args
        Unused. Reserved for future compatibility.
    **kwargs
        Plot customization. Recognized keys:
        - levels_pdf: int levels for PDF filled contour (default 35)
        - levels_contour: int levels for life contour (default 10)
        - cmap: colormap for PDF (default 'Blues')
        - title: optional title
        - inline_fontsize: fontsize for contour labels (default 8)
        - fig: optional matplotlib Figure
        - figsize: figure size if fig is not provided

    Returns
    -------
    (fig, ax)
        The created figure and axes.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> class Dummy:
    ...     def pdf(self, x):
    ...         return np.ones_like(x, dtype=float)
    >>> df = pd.DataFrame([[1., 2.], [3., 4.]],
    ...                   index=[5., 6.], columns=[0., 1.])
    >>> fig, ax = contour_life_d_mis(
    ...     df, Dummy(), Dummy(), levels_contour=3
    ... )
    >>> isinstance(fig, plt.Figure)
    True
    """
    levels_pdf = kwargs.pop("levels_pdf", 35)
    levels_contour = kwargs.pop("levels_contour", 10)
    cmap = kwargs.pop("cmap", "Blues")
    title = kwargs.pop("title", None)
    inline_fontsize = kwargs.pop("inline_fontsize", 8)
    units = kwargs.pop("units", "mm")

    fig = kwargs.pop("fig", None)
    if fig is None:
        fig = plt.figure(figsize=kwargs.pop("figsize", (16 / 2.54, 7 / 2.54)))
    if "ax" in kwargs:
        ax = kwargs.pop("ax")
    else:
        ax = fig.add_subplot(111)

    d_unique = pivot_df.index.values
    misalignment_unique = pivot_df.columns.values
    life_grid = pivot_df.values
    d_mesh, misalignment_mesh = np.meshgrid(
        d_unique, misalignment_unique, indexing="ij"
    )

    d_fine = np.linspace(d_unique.min(), d_unique.max(), 200)
    mis_fine = np.linspace(
        misalignment_unique.min(), misalignment_unique.max(), 200
    )
    d_fine_mesh, mis_fine_mesh = np.meshgrid(d_fine, mis_fine, indexing="ij")
    pdf_d = trunc_d.pdf(d_fine_mesh)
    pdf_mis = trunc_misalignment.pdf(mis_fine_mesh)
    joint_pdf = pdf_d * pdf_mis

    ax.contourf(
        d_fine_mesh, mis_fine_mesh, joint_pdf, levels=levels_pdf, cmap=cmap,
        alpha=0.6,
    )
    cs = ax.contour(
        d_mesh, misalignment_mesh, life_grid, levels=levels_contour, colors="k"
    )
    ax.clabel(cs, inline=True, fontsize=inline_fontsize, fmt="%1.1f")
    ax.set_xlabel(f"Diameter, {units}")
    ax.set_ylabel(f"Misalignment, {units}")
    if title:
        ax.set_title(title)
    return fig, ax


def life_t1_t2_mis(
    df_slice_3d: pd.DataFrame, *args: Any, **kwargs: Any
) -> tuple[Figure, "mpl_toolkits.mplot3d.axes3d.Axes3D"]:  # type: ignore
    """
    3D scatter of Fatigue_life_IN vs (t, T) colored by misalignment.

    Parameters
    ----------
    df_slice_3d
        DataFrame containing columns:
        - 't'
        - 'T'
        - 'Fatigue_life_IN'
        - 'misalignment'
    *args
        Unused. Reserved for future compatibility.
    **kwargs
        Plot customization. Recognized keys:
        - cmap: colormap for scatter (default 'coolwarm_r')
        - s: marker size (default 30)
        - title: plot title (default set)

    Returns
    -------
    (fig, ax)
        The created figure and the 3D axes.

    Examples
    --------
    >>> import pandas as pd
    >>> df3 = pd.DataFrame({
    ...     "t": [1, 2],
    ...     "T": [10, 11],
    ...     "Fatigue_life_IN": [100, 110],
    ...     "misalignment": [0.1, 0.2],
    ... })
    >>> fig, ax = life_t1_t2_mis(df3)
    >>> ax.name == "3d"
    True
    """
    fig = kwargs.pop("fig", None)
    if fig is None:
        fig = plt.figure(figsize=kwargs.pop("figsize", (10, 7)))
    ax = fig.add_subplot(111, projection="3d")

    t_vals = df_slice_3d["t"].to_numpy(dtype=float)
    t2_vals = df_slice_3d["T"].to_numpy(dtype=float)
    life_vals = df_slice_3d["Fatigue_life_IN"].to_numpy(dtype=float)
    mis_vals = df_slice_3d["misalignment"].to_numpy(dtype=float)

    p = ax.scatter(
        t_vals,
        t2_vals,
        zs=life_vals,  # type: ignore
        c=mis_vals,
        cmap=kwargs.pop("cmap", "coolwarm_r"),
        s=kwargs.pop("s", 30),
    )
    ax.set_xlabel("Thickness t")
    ax.set_ylabel("Thickness T")
    ax.set_zlabel("Fatigue Life inside")
    ax.set_title(kwargs.pop("title", "Fatigue Life inside vs t & T"))
    fig.colorbar(p, ax=ax, label="Misalignment")
    return fig, ax


def cans_interface_3d(
    t_bottom: float,
    t_top: float,
    d_outer: float,
    misalignment: float,
    length_bottom: float = 40.0,
    length_top: float = 40.0,
    n_theta: int = 80,
    n_z: int = 5,
    alpha_shell: float = 0.25,
    **kwargs: Any,
) -> tuple[Figure, "mpl_toolkits.mplot3d.axes3d.Axes3D"]:  # type: ignore
    """
    Plot a 3D view of the interface between two monopile cans.

    Parameters
    ----------
    t_bottom : float
        Bottom can thickness [mm].
    t_top : float
        Top can thickness [mm].
    d_outer : float
        Outer diameter [mm], assumed constant across the interface.
    misalignment : float
        Lateral offset between can centerlines [mm], along +x for the
        top can.
    length_bottom : float, optional
        Length of the bottom can segment to render [mm].
    length_top : float, optional
        Length of the top can segment to render [mm].
    n_theta : int, optional
        Number of circumferential points.
    n_z : int, optional
        Number of axial points per can.
    alpha_shell : float, optional
        Transparency for shell surfaces.
    **kwargs
        Additional plotting options. Recognized keys:
        - elev: view elevation (default 20)
        - azim: view azimuth (default 15)
        - fig: optional matplotlib Figure
        - figsize: figure size if fig is not provided

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        3D axes of the plot.

    Examples
    --------
    >>> fig, ax = cans_interface_3d(
    ...     t_bottom=20.0, t_top=25.0, d_outer=6000.0, misalignment=50.0,
    ...     n_theta=16, n_z=3
    ... )
    >>> isinstance(fig, plt.Figure) and ax.name == "3d"
    True
    """
    elev = kwargs.pop("elev", 20)
    azim = kwargs.pop("azim", 15)
    units = kwargs.pop("units", "mm")

    fig = kwargs.pop("fig", None)
    if fig is None:
        fig = plt.figure(figsize=kwargs.pop("figsize", (8, 6)))
    ax = fig.add_subplot(111, projection="3d")

    r_out = d_outer / 2.0
    r_in_bottom = r_out - t_bottom
    r_in_top = r_out - t_top
    offset_x = float(misalignment)

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    z_bottom = np.linspace(-float(length_bottom), 0.0, n_z)
    z_top = np.linspace(0.0, float(length_top), n_z)
    theta_bottom, z_mesh_bottom = np.meshgrid(theta, z_bottom)
    theta_top, z_mesh_top = np.meshgrid(theta, z_top)

    # Bottom can surfaces (centered at x=0, y=0)
    x_bottom_out = r_out * np.cos(theta_bottom)
    y_bottom_out = r_out * np.sin(theta_bottom)
    x_bottom_in = r_in_bottom * np.cos(theta_bottom)
    y_bottom_in = r_in_bottom * np.sin(theta_bottom)

    # Top can surfaces (shifted by offset_x in x)
    x_top_out = offset_x + r_out * np.cos(theta_top)
    y_top_out = r_out * np.sin(theta_top)
    x_top_in = offset_x + r_in_top * np.cos(theta_top)
    y_top_in = r_in_top * np.sin(theta_top)

    # Draw shells
    ax.plot_surface(
        x_bottom_out, y_bottom_out, z_mesh_bottom, color="C0",
        alpha=alpha_shell, lw=0
    )
    ax.plot_surface(
        x_bottom_in, y_bottom_in, z_mesh_bottom, color="C0",
        alpha=alpha_shell * 0.6, lw=0
    )
    ax.plot_surface(
        x_top_out, y_top_out, z_mesh_top, color="C1",
        alpha=alpha_shell, lw=0
    )
    ax.plot_surface(
        x_top_in, y_top_in, z_mesh_top, color="C1",
        alpha=alpha_shell * 0.6, lw=0
    )

    # Interface rings at z=0
    th = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(
        r_out * np.cos(th), r_out * np.sin(th), 0.0 * th,
        color="C2", lw=0.5, ls="--", label="outer ring (bottom)"
    )
    ax.plot(
        offset_x + r_out * np.cos(th), r_out * np.sin(th), 0.0 * th,
        color="C3", lw=0.5, ls="--", label="outer ring (top)"
    )
    ax.plot(
        r_in_bottom * np.cos(th), r_in_bottom * np.sin(th), 0.0 * th,
        color="C2", lw=0.5, label="inner ring (bottom)"
    )
    ax.plot(
        offset_x + r_in_top * np.cos(th), r_in_top * np.sin(th), 0.0 * th,
        color="C3", lw=0.5, label="inner ring (top)"
    )

    # Misalignment arrow
    ax.quiver(
        0.0, 0.0, 0.0, offset_x, 0.0, 0.0, color="r", lw=1,
        arrow_length_ratio=0.05
    )
    ax.text(offset_x * 0.75, 0.0, 0.0, "misalignment", color="r")

    # Thickness arrow bottom (at theta=90 deg plane)
    z_b_annot = -0.6 * float(length_bottom)
    ax.quiver(
        0.0, r_in_bottom, z_b_annot, 0.0, t_bottom, 0.0, color="C0", lw=1,
        arrow_length_ratio=0.5
    )
    ax.text(
        0.0, r_in_bottom + t_bottom * 1.55, z_b_annot,
        f"t={t_bottom:.1f} mm", color="C0", va="center"
    )

    # Thickness arrow top (at theta=0 deg plane)
    z_t_annot = 0.6 * float(length_top)
    ax.quiver(
        offset_x + r_in_top, 0.0, z_t_annot, t_top, 0.0, 0.0, color="C1",
        lw=1, arrow_length_ratio=0.05
    )
    ax.text(
        offset_x + r_in_top + t_top * 0.55, 0.0 + t_top * 0.55, z_t_annot,
        f"T={t_top:.1f} mm", color="C1", va="center"
    )

    # Axes labels and view
    ax.set_xlabel(f"x, {units}")
    ax.set_ylabel(f"y, {units}")
    ax.set_title("Monopile can interface (outer diameter constant)")
    ax.legend(loc="upper right", fontsize=7)

    # Fit limits
    reach_r = r_out + abs(offset_x) + 0.1 * r_out
    ax.set_xlim(-reach_r, reach_r)
    ax.set_ylim(-reach_r, reach_r)
    ax.set_zlim(-float(length_bottom * 5), float(length_top * 5))
    ax.view_init(elev, azim)

    ax.set_aspect("equal")

    # Remove z-axis ticks and labels
    ax.set_zticks([])  # type: ignore
    ax.set_zticklabels([])  # type: ignore

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["tick"]["inward_factor"] = 0  # type: ignore
        axis._axinfo["tick"]["outward_factor"] = 0  # type: ignore
    return fig, ax

def misalignment_surfaces_t1_t2_by_fatigue_life_at_d(
    df: pd.DataFrame,
    d_fix: float,
    *args: Any,
    tol: float = 1e-5,
    n_surfaces: int = 10,
    **kwargs: Any,
) -> tuple[Figure, "mpl_toolkits.mplot3d.axes3d.Axes3D"]:  # type: ignore
    """
    Plot 3D life surfaces vs t and T across misalignment levels at fixed D.

    The function slices the input DataFrame at a fixed diameter D (within a
    tolerance), selects up to n_surfaces misalignment levels, and draws one
    surface per level: z=Fatigue_life_IN over the (t, T) grid. A colorbar
    encodes the misalignment value used for each surface.

    Parameters
    ----------
    df
        Input DataFrame containing columns:
        - 'D' (diameter),
        - 't', 'T' (thicknesses),
        - 'Fatigue_life_IN' (life),
        - 'misalignment'.
    d_fix
        Diameter value used to slice the DataFrame.
    tol
        Absolute tolerance for selecting rows with D ~= d_fix.
    n_surfaces
        Maximum number of misalignment surfaces to draw (default 10).
    *args
        Unused. Reserved for future compatibility.
    **kwargs
        Plot customization. Recognized keys:
        - cmap: colormap or name (default 'coolwarm_r').
        - alpha: surface alpha (default 0.75).
        - edgecolor: surface edge color (default 'none').
        - title: optional title (default set).
        - fig: optional matplotlib Figure.
        - figsize: figure size if fig is not provided.
        - azim: view azimuth (default 75).
        - elev: view elevation (default 20).
        - units: units string for labels (default 'mm').
        - axes_ratios: aspect ratios for (x, y, z) axes (default (1, 1, 0.5)).
        - cmap_position: colorbar position [left, bottom, width, height]
          (default [0.275, 0., 0.5, 0.03]).

    Returns
    -------
    (fig, ax)
        The created figure and the 3D axes.

    Raises
    ------
    ValueError
        If no data rows match the requested diameter slice.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df_ex = pd.DataFrame({
    ...     "D": [100., 100., 100., 100.] * 2,
    ...     "t": [1., 1., 2., 2.] * 2,
    ...     "T": [10., 11., 10., 11.] * 2,
    ...     "Fatigue_life_IN": np.arange(8, dtype=float),
    ...     "misalignment": [0., 0., 0., 0., 1., 1., 1., 1.],
    ... })
    >>> fig, ax = life_surfaces_t1_t2_by_misalignment_at_d(
    ...     df_ex, d_fix=100., tol=1e-9, n_surfaces=2
    ... )
    >>> ax.name == "3d"
    True
    >>> len(ax.collections)
    2
    """
    # kwargs with defaults
    cmap_in = kwargs.pop("cmap", "coolwarm_r")
    alpha = float(kwargs.pop("alpha", 0.75))
    edgecolor = kwargs.pop("edgecolor", "none")
    linewidth = float(kwargs.pop("linewidth", 0.0))
    if edgecolor != "none" and linewidth <= 0.0:
        linewidth = 0.5
    if edgecolor == "none":
        linewidth = 0.0
    kwargs["linewidth"] = linewidth  # pass to plot_surface
    elev = kwargs.pop("elev", 15)
    azim = kwargs.pop("azim", 90)
    units = kwargs.pop("units", "mm")
    title = kwargs.pop("title", f"Fatigue Life surfaces vs t & T at D={d_fix}")
    fig = kwargs.pop("fig", None)
    figsize = kwargs.pop("figsize", (10, 7))
    axes_ratios = kwargs.pop("axes_ratios", (1, 1, 0.5))
    cmap_position = kwargs.pop("cmap_position", [0.275, 0., 0.5, 0.03])
    aspect_zoom = kwargs.pop("aspect_zoom", 1.0)

    # Slice at fixed diameter
    df_slice_3d = df[np.isclose(df["D"], d_fix, atol=tol)]
    if df_slice_3d.empty:
        raise ValueError(
            "No rows found matching the requested diameter slice."
        )

    # Choose a manageable subset of misalignment levels for clarity
    mis_unique = np.sort(df_slice_3d["misalignment"].unique())
    if mis_unique.size > n_surfaces:
        idx = np.linspace(0, mis_unique.size - 1, n_surfaces)
        idx = idx.round().astype(int)
        mis_levels = mis_unique[idx]
    else:
        mis_levels = mis_unique

    # Colormap and normalization
    cmap = plt.cm.get_cmap(cmap_in) if isinstance(cmap_in, str) else cmap_in
    norm = Normalize(vmin=mis_unique.min(), vmax=mis_unique.max())

    # Figure and axes
    if fig is None:
        fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    # Draw one surface per misalignment level
    for mis in mis_levels:
        sub = df_slice_3d[
            np.isclose(df_slice_3d["misalignment"], mis, atol=tol)
        ]
        grid = (
            sub.pivot_table(index="t", columns="T", values="Fatigue_life_IN")
            .sort_index()
            .sort_index(axis=1)
        )
        if grid.empty:
            continue
        t_grid_vals = grid.index.values
        T_grid_vals = grid.columns.values
        T_grid, t_grid = np.meshgrid(T_grid_vals, t_grid_vals)
        color = cmap(norm(mis))
        ax.plot_surface(
            t_grid,
            T_grid,
            grid.values,
            color=color,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
            shade=False
        )

    # Axes labels and view
    ax.set_xlabel(f"Thickness t, {units}")
    ax.set_ylabel(f"Thickness T, {units}")
    ax.set_zlabel("Fatigue Life Inside, years")
    ax.view_init(elev, azim)
    # Colorbar keyed to misalignment
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cax = fig.add_axes(cmap_position)  # type: ignore
    fig.colorbar(mappable, cax=cax, label=f"Misalignment, {units}",
                 orientation="horizontal")
    if title:
        ax.set_title(title)

    ax.set_box_aspect(axes_ratios)  # Different aspect ratio for z
    ax.set_position([0, 0, 1, 1])  # type: ignore
    ax.set_box_aspect(None, zoom=aspect_zoom)  # type: ignore
    # Hide only the tick marks, keep labels
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["tick"]["inward_factor"] = 0  # type: ignore
        axis._axinfo["tick"]["outward_factor"] = 0  # type: ignore

    return fig, ax


def life_surfaces_t1_t2_by_misalignment_at_d(
    df: pd.DataFrame,
    d_fix: float,
    *args: Any,
    tol: float = 3,
    n_surfaces: int = 10,
    **kwargs: Any,
) -> tuple[Figure, "mpl_toolkits.mplot3d.axes3d.Axes3D"]:  # type: ignore
    """
    Plot 3D life surfaces vs t and T across misalignment levels at fixed D.

    The function slices the input DataFrame at a fixed diameter D (within a
    tolerance), selects up to n_surfaces misalignment levels, and draws one
    surface per level: z=Fatigue_life_IN over the (t, T) grid. A colorbar
    encodes the misalignment value used for each surface.

    Parameters
    ----------
    df
        Input DataFrame containing columns:
        - 'D' (diameter),
        - 't', 'T' (thicknesses),
        - 'Fatigue_life_IN' (life),
        - 'misalignment'.
    d_fix
        Diameter value used to slice the DataFrame.
    tol
        Absolute tolerance for selecting rows with D ~= d_fix.
    n_surfaces
        Maximum number of misalignment surfaces to draw (default 10).
    *args
        Unused. Reserved for future compatibility.
    **kwargs
        Plot customization. Recognized keys:
        - cmap: colormap or name (default 'coolwarm_r').
        - alpha: surface alpha (default 0.75).
        - edgecolor: surface edge color (default 'none').
        - title: optional title (default set).
        - fig: optional matplotlib Figure.
        - figsize: figure size if fig is not provided.
        - azim: view azimuth (default 75).
        - elev: view elevation (default 20).
        - units: units string for labels (default 'mm').

    Returns
    -------
    (fig, ax)
        The created figure and the 3D axes.

    Raises
    ------
    ValueError
        If no data rows match the requested diameter slice.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df_ex = pd.DataFrame({
    ...     "D": [100., 100., 100., 100.] * 2,
    ...     "t": [1., 1., 2., 2.] * 2,
    ...     "T": [10., 11., 10., 11.] * 2,
    ...     "Fatigue_life_IN": np.arange(8, dtype=float),
    ...     "misalignment": [0., 0., 0., 0., 1., 1., 1., 1.],
    ... })
    >>> fig, ax = life_surfaces_t1_t2_by_misalignment_at_d(
    ...     df_ex, d_fix=100., tol=1e-9, n_surfaces=2
    ... )
    >>> ax.name == "3d"
    True
    >>> len(ax.collections)
    2
    """
    # kwargs with defaults
    cmap_in = kwargs.pop("cmap", "coolwarm_r")
    alpha = float(kwargs.pop("alpha", 0.5))
    edgecolor = kwargs.pop("edgecolor", "none")
    linewidth = float(kwargs.pop("linewidth", 0.0))
    if edgecolor != "none" and linewidth <= 0.0:
        linewidth = 0.5
    if edgecolor == "none":
        linewidth = 0.0
    kwargs["linewidth"] = linewidth  # pass to plot_surface
    elev = kwargs.pop("elev", 20)
    azim = kwargs.pop("azim", 290)
    units = kwargs.pop("units", "mm")
    title = kwargs.pop("title", None)
    axes_ratios = kwargs.pop("axes_ratios", (1, 1, 0.25))
    fig = kwargs.pop("fig", None)
    figsize = kwargs.pop("figsize", (10, 7))
    fat_column = kwargs.pop("fat_column", "Fatigue_life_IN")
    # Slice at fixed diameter
    df_slice_3d = df[np.isclose(df["D"], d_fix, atol=tol)]
    if df_slice_3d.empty:
        raise ValueError(
            "No rows found matching the requested diameter slice."
        )

    # Choose a manageable subset of fatigue life levels for clarity
    fat_unique = np.sort(df_slice_3d[fat_column].unique())
    if fat_unique.size > n_surfaces:
        idx = np.linspace(0, fat_unique.size - 1, n_surfaces)
        idx = idx.round().astype(int)
        fat_levels = fat_unique[idx]
    else:
        fat_levels = fat_unique
    # logger.debug(" 🪳 fat_levels = %s", fat_levels)
    # Colormap and normalization
    cmap = plt.cm.get_cmap(cmap_in) if isinstance(cmap_in, str) else cmap_in
    norm = Normalize(vmin=fat_unique.min(), vmax=fat_unique.max())
    # Figure and axes
    if fig is None:
        fig = plt.figure(figsize=figsize)
    if "ax" in kwargs:
        ax = kwargs.pop("ax")
    else:
        ax = fig.add_subplot(111, projection="3d")
    # Draw one surface per misalignment level
    for fat in fat_levels:
        logger.debug(" 🪳 fat = %s", fat)
        sub = df_slice_3d[
            np.isclose(df_slice_3d[fat_column], fat, atol=tol)
        ]
        logger.debug("    • sub.shape = %s", sub.shape)
        grid = (
            sub.pivot_table(index="t", columns="T", values="misalignment")
            .sort_index()
            .sort_index(axis=1)
        )
        logger.debug("    • grid.shape = %s", grid.shape)
        if grid.empty:
            continue
        t_grid_vals = grid.index.values
        T_grid_vals = grid.columns.values
        logger.debug("    • t_grid_vals = %s", t_grid_vals)
        logger.debug("    • T_grid_vals = %s", T_grid_vals)
        T_grid, t_grid = np.meshgrid(T_grid_vals, t_grid_vals)
        logger.debug("    • T_grid.shape = %s", T_grid.shape)
        color = cmap(norm(fat))
        logger.debug("    • color = %s", color)
        # fmt: off
        ax.plot_surface(t_grid, T_grid, grid.values, color=color, alpha=alpha,
                        edgecolor=edgecolor, linewidth=linewidth, shade=False)
        # fmt: on
    # Axes labels and view
    ax.set_xlabel(f"Thickness t, {units}")
    ax.set_ylabel(f"Thickness T, {units}")
    ax.set_zlabel(f"Misalignment, {units}")
    ax.view_init(elev, azim)
    # Colorbar keyed to misalignment
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cax = fig.add_axes([0.275, 0.125, 0.5, 0.03])  # type: ignore
    fig.colorbar(mappable, cax=cax, label="Fatigue Life Inside, years",
                 orientation="horizontal")
    if title:
        ax.set_title(title)

    ax.set_box_aspect(axes_ratios)  # Different aspect ratio for z
    ax.set_position([0, 0, 1, 1])  # type: ignore
    # Hide only the tick marks, keep labels
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["tick"]["inward_factor"] = 0  # type: ignore
        axis._axinfo["tick"]["outward_factor"] = 0  # type: ignore
    return fig, ax
