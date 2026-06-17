# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Shared engine for the variance-map themes.

What a "theme" is
-----------------
A *theme* is **a type of information source and the uncertainty model attached
to it** — not a data source and not a particular dataset. An information source
is anything that constrains where a layer boundary sits: a geophysical survey, a
borehole, a geological interpretation, a digitised cross-section, and so on. The
engine never touches the raw data. It is fed only:

* the **x, y locations** where a source of one type informs the boundary, and
* the per-location **attributes** that govern the uncertainty there — the
  interpreted boundary *depth* at that location, the layer *thickness*, the
  source's *depth-of-investigation*, etc.

From those, plus the *distance* from each grid cell to the nearest informed
location, it computes the **variance of interpreting that layer boundary's depth
given this type of information**. The deeper the boundary and the farther a cell
is from an informed location, the larger the variance — exactly how that grows
is the theme's parameterisation. The built-in themes happen to be geophysical
(PACES, GAMMALOG, RESLOG, REFSEIS, fewTEM, manyTEM, tTEM), but that is
incidental: a borehole or geological-interpretation theme is just another spec.

The algorithm (the same for every theme, only the parameters differ)
--------------------------------------------------------------------
1. **Windowed nearest-source pass.** Slide a ``±search_radius``-cell window over
   the grid; within it pick the single *nearest* informed location and record
   its distance plus that location's attributes (interpreted depth, thickness,
   depth-of-investigation), per layer. Where several locations sit at exactly the
   minimum distance, attributes are reduced over them (mean / min / max,
   theme-dependent).
2. **Complexity -> range.** A geological-complexity class grid is mapped to an
   effective correlation range, with a "prerequisite layer" override.
3. **Certainty -> variance.** ``cert = cert_fun(dist, range, kernel, 1/var0)``
   then ``variance = 1 / cert``; cells with no informing location get the
   :data:`NODATA_VARIANCE` sentinel, as do cells failing the theme's depth/DOI
   mask (e.g. the boundary lies below the source's depth-of-investigation).

Two knobs, two source archetypes
--------------------------------
The whole per-source behaviour is captured by two callables on the spec, and
between them they cover the common cases:

* ``mask_fn`` — *can this source see the boundary at all?* When the boundary
  depth exceeds what the source reached, the source carries no information and
  the cell gets :data:`NODATA_VARIANCE`. For a **borehole / well** this is the
  borehole's total depth: a layer deeper than the borehole is maximally
  uncertain from that well (``doi < depth``). For geophysics it is the
  method's depth-of-investigation.
* ``var0_fn`` — *how certain is the interpretation where the source does see
  the boundary?* Some methods have sensitivity that is roughly **constant with
  depth** (a flat ``var0``, e.g. reflection seismic); others **degrade with
  depth** (``var0`` grows with the boundary depth, e.g. PACES and the TEM
  methods). Either is just the formula chosen in the spec.

This module holds the generic, **source-agnostic, model-agnostic** machinery.
The per-theme numbers (search radius, range tables, the ``var0`` / ``kernel`` /
mask formulae, the certainty kernel) live in :mod:`geosigma.themes.themes` as
:class:`ThemeSpec` objects. Reading a particular source's data into locations +
attributes (e.g. the Danish ``.mat`` files, their column names, region
selection) is the job of *adapters* outside the library (see ``examples/``); the
engine only ever sees plain arrays.

Vectorisation
-------------
The original MATLAB nests two loops over grid cells (``ny * nx`` iterations).
Because the nearest-point assignment depends only on the *data-point* positions
— not on the layer — we instead loop over the (far fewer) **data points** and
scatter each point's contribution onto the block of grid cells whose search
window contains it. This removes the nested grid-cell loops entirely while
reproducing the window logic exactly; ``tests/test_themes.py`` asserts the
vectorised result is identical to a direct triple-loop port of the MATLAB code.

Coordinate convention
----------------------
``grid_x`` / ``grid_y`` are 1-D, **ascending** coordinate axes (east / north),
matching the MATLAB ``UTM_X`` / ``UTM_Y`` vectors. Returned grids are indexed
``[iy, ix]`` so that element ``[iy, ix]`` sits at ``(grid_x[ix], grid_y[iy])``.

Omitted: peatland / NPL
-----------------------
The MATLAB themes carry a peat-layer count ``NPL`` that (a) offsets the layer
origin and (b) prepends ``NPL`` constant-variance layers when
``include_peatlands`` is set. Both are intentionally dropped here (out of scope,
see ``CLAUDE.md``); the engine works purely in modelled-layer space
``0 .. n_layers-1``. Re-adding peat is an isolated wrapper that prepends constant
layers around this output — no engine change needed. The peat *content*
(``get_PL_themes``) is excluded entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

#: Variance sentinel for "no information" cells (the MATLAB ``100000``).
NODATA_VARIANCE = 100000.0


@dataclass
class ThemeData:
    """Model-agnostic inputs to a theme builder.

    Parameters
    ----------
    xs, ys : ndarray, shape (npoints,)
        Data-point coordinates (east, north), already deduplicated / date-filtered
        by the adapter.
    attributes : mapping of str -> ndarray, shape (npoints, n_layers)
        Per-point, per-layer model attributes the theme needs (e.g. ``"depth"``,
        ``"thick"``, ``"doi"``). A 1-D ``(npoints,)`` array is broadcast across
        layers. Which attributes are required is declared by the
        :class:`ThemeSpec`.
    complexity : ndarray, shape (ny, nx)
        Geological-complexity class grid (integer classes; ``0`` = unknown ->
        masked). Indexed ``[iy, ix]``.
    """

    xs: np.ndarray
    ys: np.ndarray
    attributes: Mapping[str, np.ndarray]
    complexity: np.ndarray


@dataclass
class ThemeSpec:
    """Per-theme parameters and formulae (the only thing that differs per theme).

    The three callables receive a mapping ``attrs`` of reduced attribute grids
    (each ``(ny, nx, n_layers)``) and must return arrays broadcastable to
    ``(ny, nx, n_layers)``. They encode the theme's ``var0``/``kernel``/mask
    formulae *verbatim* from the MATLAB source — the statistical model is never
    altered here.
    """

    name: str
    search_radius: int
    comp2range: Sequence[float]
    comp2range_preq: Sequence[float]
    n_prereq: int
    cert_fun_name: str
    attributes: Sequence[str]
    reducers: Mapping[str, str]
    var0_fn: Callable[[Mapping[str, np.ndarray]], np.ndarray]
    kernel_fn: Callable[[Mapping[str, np.ndarray]], np.ndarray]
    mask_fn: Callable[[Mapping[str, np.ndarray]], np.ndarray]
    #: Optional subset of layer indices that receive data; others -> NODATA.
    active_layers: Optional[Sequence[int]] = None


_REDUCERS = ("mean", "min", "max")


def _broadcast_attr(arr: np.ndarray, n_layers: int) -> np.ndarray:
    """Return ``arr`` shaped ``(npoints, n_layers)``, broadcasting a 1-D array."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = np.repeat(arr[:, None], n_layers, axis=1)
    if arr.shape[1] != n_layers:
        raise ValueError(
            f"attribute has {arr.shape[1]} layer columns, expected {n_layers}"
        )
    return arr


def windowed_nearest(
    xs: np.ndarray,
    ys: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    search_radius: int,
    attributes: Mapping[str, np.ndarray],
    reducers: Mapping[str, str],
    n_layers: int,
):
    """Vectorised windowed nearest-point pass.

    For every grid cell, find the nearest data point lying strictly inside the
    ``±search_radius``-cell coordinate window, and reduce the requested
    attributes over all points at exactly that minimum distance.

    Returns
    -------
    local_dist : ndarray, shape (ny, nx)
        Distance to the nearest in-window data point; ``NaN`` where the window
        held no points. (Layer-independent — the same for every layer.)
    reduced : dict of str -> ndarray, shape (ny, nx, n_layers)
        Each requested attribute, reduced over the minimum-distance ties.
        ``NaN`` where no point was found.

    Notes
    -----
    Loops over data points (not grid cells); each point scatters onto the block
    of cells whose window contains it. Distances use the same ``sqrt`` form and
    exact tie comparison as the MATLAB original, so the minimum-distance tie set
    matches bit-for-bit.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    grid_x = np.asarray(grid_x, dtype=float)
    grid_y = np.asarray(grid_y, dtype=float)
    nx = grid_x.size
    ny = grid_y.size
    R = int(search_radius)

    for name in attributes:
        if reducers.get(name) not in _REDUCERS:
            raise ValueError(
                f"attribute {name!r} needs a reducer in {_REDUCERS}, "
                f"got {reducers.get(name)!r}"
            )
    attrs = {n: _broadcast_attr(attributes[n], n_layers) for n in attributes}

    # Per-cell window bounds (open interval), with edge clamping, as in MATLAB:
    #   ymin = UTM_Y[max(0, i-R)] ; ymax = UTM_Y[min(ny-1, i+R)]
    idx_y = np.arange(ny)
    idx_x = np.arange(nx)
    ylo = grid_y[np.maximum(0, idx_y - R)]
    yhi = grid_y[np.minimum(ny - 1, idx_y + R)]
    xlo = grid_x[np.maximum(0, idx_x - R)]
    xhi = grid_x[np.minimum(nx - 1, idx_x + R)]

    # ---- Pass 1: minimum in-window distance per cell -------------------------
    best = np.full((ny, nx), np.inf)
    npoints = xs.size
    for p in range(npoints):
        px, py = xs[p], ys[p]
        rows = np.nonzero((ylo < py) & (py < yhi))[0]
        if rows.size == 0:
            continue
        cols = np.nonzero((xlo < px) & (px < xhi))[0]
        if cols.size == 0:
            continue
        d = np.sqrt(
            (grid_x[cols][None, :] - px) ** 2 + (grid_y[rows][:, None] - py) ** 2
        )
        block = np.ix_(rows, cols)
        best[block] = np.minimum(best[block], d)

    local_dist = np.where(np.isfinite(best), best, np.nan)

    # ---- Pass 2: reduce attributes over minimum-distance ties ----------------
    count = np.zeros((ny, nx), dtype=np.int64)
    accum = {}
    for name in attrs:
        red = reducers[name]
        if red == "mean":
            accum[name] = np.zeros((ny, nx, n_layers))
        elif red == "min":
            accum[name] = np.full((ny, nx, n_layers), np.inf)
        else:  # max
            accum[name] = np.full((ny, nx, n_layers), -np.inf)

    for p in range(npoints):
        px, py = xs[p], ys[p]
        rows = np.nonzero((ylo < py) & (py < yhi))[0]
        if rows.size == 0:
            continue
        cols = np.nonzero((xlo < px) & (px < xhi))[0]
        if cols.size == 0:
            continue
        d = np.sqrt(
            (grid_x[cols][None, :] - px) ** 2 + (grid_y[rows][:, None] - py) ** 2
        )
        tie = d == best[np.ix_(rows, cols)]
        if not tie.any():
            continue
        ti, tj = np.nonzero(tie)
        gi = rows[ti]
        gj = cols[tj]
        np.add.at(count, (gi, gj), 1)
        for name in attrs:
            vals = attrs[name][p]  # (n_layers,)
            red = reducers[name]
            if red == "mean":
                np.add.at(accum[name], (gi, gj), vals)
            elif red == "min":
                np.minimum.at(accum[name], (gi, gj), vals)
            else:
                np.maximum.at(accum[name], (gi, gj), vals)

    has = count > 0
    reduced = {}
    for name in attrs:
        red = reducers[name]
        out = np.full((ny, nx, n_layers), np.nan)
        if red == "mean":
            np.divide(accum[name], count[:, :, None], out=out, where=has[:, :, None])
        else:
            out[has] = accum[name][has]
        reduced[name] = out

    return local_dist, reduced


def _range_map(complexity, comp2range, comp2range_preq, n_prereq, n_layers):
    """Build the per-layer effective-range grid from a complexity class grid.

    Class ``0`` (unknown) -> ``NaN`` (cell ends up masked). Classes ``1..4`` map
    through ``comp2range``; layers from ``n_prereq`` onward use
    ``comp2range_preq`` instead. Verbatim from the MATLAB ``rangemap`` logic.
    """
    complexity = np.asarray(complexity, dtype=float)
    base = np.full(complexity.shape, np.nan)
    preq = np.full(complexity.shape, np.nan)
    for c in range(1, 5):
        sel = complexity == c
        base[sel] = comp2range[c]
        preq[sel] = comp2range_preq[c]
    rangemap = np.repeat(base[:, :, None], n_layers, axis=2)
    if n_prereq < n_layers:
        rangemap[:, :, n_prereq:] = preq[:, :, None]
    return rangemap


def build_theme(
    data: ThemeData,
    spec: ThemeSpec,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    n_layers: int,
    window_fn=windowed_nearest,
):
    """Build a per-layer variance grid for one information-source theme.

    Parameters
    ----------
    window_fn : callable, optional
        The windowed nearest-source pass, defaulting to the vectorised
        :func:`windowed_nearest`. Exposed as a seam so tests can substitute a
        direct triple-loop reference and assert the full pipeline is identical
        under either windowing implementation.

    Returns
    -------
    ndarray, shape (ny, nx, n_layers)
        The theme's variance map; :data:`NODATA_VARIANCE` where the theme carries
        no usable information (no informing location in window, unknown
        complexity, or the theme's depth/DOI mask).
    """
    grid_x = np.asarray(grid_x, dtype=float)
    grid_y = np.asarray(grid_y, dtype=float)

    local_dist, attrs = window_fn(
        data.xs,
        data.ys,
        grid_x,
        grid_y,
        spec.search_radius,
        {n: data.attributes[n] for n in spec.attributes},
        spec.reducers,
        n_layers,
    )

    rangemap = _range_map(
        data.complexity,
        spec.comp2range,
        spec.comp2range_preq,
        spec.n_prereq,
        n_layers,
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        var0map = spec.var0_fn(attrs)
        kernelmap = spec.kernel_fn(attrs)
        from .certainty_functions import certainty_function

        cert_fun = certainty_function(spec.cert_fun_name)
        dist3 = np.repeat(local_dist[:, :, None], n_layers, axis=2)
        certmap = cert_fun(dist3, rangemap, kernelmap, 1.0 / var0map)
        grid = 1.0 / certmap

    grid[np.isnan(grid)] = NODATA_VARIANCE
    grid[spec.mask_fn(attrs)] = NODATA_VARIANCE

    if spec.active_layers is not None:
        active = np.zeros(n_layers, dtype=bool)
        active[np.asarray(spec.active_layers, dtype=int)] = True
        grid[:, :, ~active] = NODATA_VARIANCE

    return grid
