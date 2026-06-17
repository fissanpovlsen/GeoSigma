# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""The built-in geophysical theme specifications.

Each function returns a :class:`~geosigma.themes.base.ThemeSpec` carrying one
theme's parameters and its ``var0`` / ``kernel`` / mask formulae, transcribed
**verbatim** from the corresponding root-level ``get_*_theme.py`` (themselves
direct MATLAB translations). The generic engine in
:mod:`geosigma.themes.base` consumes these specs; the Danish ``.mat`` loading
that feeds them lives in adapters outside the library (see ``examples/``).

Layer indexing here is modelled-layer space (``0 .. n_layers-1``); the peat
offset ``NPL`` is dropped (see :mod:`geosigma.themes.base`). ``n_prereq`` is the
modelled-layer index from which the "prerequisite-layer" range override applies
(the MATLAB ``Npreq - NPL``).

The DOI ("depth of investigation") values are supplied by the adapter already
resolved: PACES overwrites DOI to a constant 18 m, and the TEM themes estimate
missing DOI from region-specific Palaeogene marker layers — that estimation is
data preparation and belongs in the adapter, not in these model-agnostic specs.
"""

from __future__ import annotations

import numpy as np

from .base import ThemeSpec

# Complexity-class -> range tables shared by most themes. Index 0 is the
# "unknown" slot (never used: complexity 0 is masked to NaN range); indices
# 1..4 are complexity classes low..high.
_COMP2RANGE = [100, 500, 400, 250, 100]
_COMP2RANGE_PREQ = [500, 500, 500, 500, 500]

# tTEM is the exception: its MATLAB source (``get_tTEM_theme.m``) uses a *flat*
# range table — every complexity class (and the prerequisite override) maps to a
# 100 m range, not the graded table above. Kept distinct so tTEM stays
# bit-faithful to the MATLAB output.
_COMP2RANGE_TTEM = [100, 100, 100, 100, 100]
_COMP2RANGE_PREQ_TTEM = [100, 100, 100, 100, 100]


def paces_spec(n_prereq: int) -> ThemeSpec:
    """PACES theme (search radius 1 cell; FRAFA plateau kernel)."""
    return ThemeSpec(
        name="PACES",
        search_radius=1,
        comp2range=_COMP2RANGE,
        comp2range_preq=_COMP2RANGE_PREQ,
        n_prereq=n_prereq,
        cert_fun_name="FRAFA_apr2023",
        attributes=("depth",),
        reducers={"depth": "mean"},
        # var0 = (0.5 * max(2, 0.25*depth))^2 ; kernel = 75 ; DOI fixed at 18.
        var0_fn=lambda a: (0.5 * np.maximum(2.0, 0.25 * a["depth"])) ** 2,
        kernel_fn=lambda a: np.full_like(a["depth"], 75.0),
        mask_fn=lambda a: 18.0 < a["depth"],
    )


def gammalog_spec(n_prereq: int) -> ThemeSpec:
    """Gamma-log theme (search radius 6; ILM Gaussian kernel; min thickness)."""
    return ThemeSpec(
        name="GAMMALOG",
        search_radius=6,
        comp2range=_COMP2RANGE,
        comp2range_preq=_COMP2RANGE_PREQ,
        n_prereq=n_prereq,
        cert_fun_name="ILM_sep2023",
        attributes=("depth", "thick"),
        reducers={"depth": "mean", "thick": "min"},
        var0_fn=_gammalog_var0,
        kernel_fn=lambda a: np.full_like(a["depth"], 150.0),
        mask_fn=lambda a: a["thick"] < a["depth"],
    )


def _gammalog_var0(a):
    # var0 = (0.5*3)^2 = 2.25 everywhere; sentinel 100000 where depth < 3.
    var0 = (0.5 * 3.0 + a["depth"] * 0.0) ** 2
    var0[a["depth"] < 3.0] = 100000.0
    return var0


def reslog_spec(n_prereq: int) -> ThemeSpec:
    """Resistivity-log theme (search radius 6; ILM Gaussian kernel; min thickness)."""
    return ThemeSpec(
        name="RESLOG",
        search_radius=6,
        comp2range=_COMP2RANGE,
        comp2range_preq=_COMP2RANGE_PREQ,
        n_prereq=n_prereq,
        cert_fun_name="ILM_sep2023",
        attributes=("depth", "thick"),
        reducers={"depth": "mean", "thick": "min"},
        var0_fn=_reslog_var0,
        kernel_fn=lambda a: np.full_like(a["depth"], 150.0),
        mask_fn=lambda a: a["thick"] < a["depth"],
    )


def _reslog_var0(a):
    # var0 = 0.25 + (0.5*0.09*depth)^2 ; sentinel 100000 where depth < 10.
    var0 = 0.25 + (0.5 * 0.09 * a["depth"]) ** 2
    var0[a["depth"] < 10.0] = 100000.0
    return var0


def refseis_spec(n_prereq: int, active_layers) -> ThemeSpec:
    """Reflection-seismic theme (search radius 8; FRAFA plateau; thickest layers).

    ``active_layers`` are the modelled-layer indices the seismic constrains (the
    MATLAB ``layvec``: layers whose geophysical thickness exceeds 1000 m). All
    other layers receive the NODATA variance.
    """
    return ThemeSpec(
        name="REFSEIS",
        search_radius=8,
        comp2range=_COMP2RANGE,
        comp2range_preq=_COMP2RANGE_PREQ,
        n_prereq=n_prereq,
        cert_fun_name="FRAFA_apr2023",
        attributes=("thick",),
        reducers={"thick": "max"},
        # var0 = (0.5*15)^2 = 56.25 constant ; kernel = 1 ; mask thin layers.
        var0_fn=lambda a: np.full_like(a["thick"], (0.5 * 15.0) ** 2),
        kernel_fn=lambda a: np.full_like(a["thick"], 1.0),
        mask_fn=lambda a: a["thick"] < 100.0,
        active_layers=active_layers,
    )


def fewtem_spec(n_prereq: int) -> ThemeSpec:
    """Few-layer TEM theme (search radius 6; ILM Gaussian kernel)."""
    return ThemeSpec(
        name="fewTEM",
        search_radius=6,
        comp2range=_COMP2RANGE,
        comp2range_preq=_COMP2RANGE_PREQ,
        n_prereq=n_prereq,
        cert_fun_name="ILM_sep2023",
        attributes=("depth", "doi"),
        reducers={"depth": "mean", "doi": "mean"},
        var0_fn=_fewtem_var0,
        kernel_fn=lambda a: np.maximum(a["depth"], 75.0),
        mask_fn=lambda a: a["doi"] < a["depth"],
    )


def _fewtem_var0(a):
    # var0 = (max(1.2, 0.5*0.15*depth) + 2)^2 ; sentinel 10000 where depth < 10.
    var0 = (np.maximum(1.2, 0.5 * 0.15 * a["depth"]) + 2.0) ** 2
    var0[a["depth"] < 10.0] = 10000.0
    return var0


def manytem_spec(n_prereq: int) -> ThemeSpec:
    """Many-layer TEM theme (search radius 6; ILM Gaussian kernel)."""
    return ThemeSpec(
        name="manyTEM",
        search_radius=6,
        comp2range=_COMP2RANGE,
        comp2range_preq=_COMP2RANGE_PREQ,
        n_prereq=n_prereq,
        cert_fun_name="ILM_sep2023",
        attributes=("depth", "thick", "doi"),
        reducers={"depth": "mean", "thick": "mean", "doi": "mean"},
        var0_fn=_manytem_var0,
        kernel_fn=lambda a: np.maximum(a["depth"], 75.0),
        mask_fn=lambda a: a["doi"] < a["depth"],
    )


def _manytem_var0(a):
    # var0 = (0.5*1.2*thick)^2 ; sentinel 100000 where depth < 7.
    var0 = (0.5 * 1.2 * a["thick"]) ** 2
    var0[a["depth"] < 7.0] = 100000.0
    return var0


def ttem_spec(n_prereq: int) -> ThemeSpec:
    """Towed TEM (tTEM) theme (search radius 6; ILM Gaussian kernel)."""
    return ThemeSpec(
        name="tTEM",
        search_radius=6,
        comp2range=_COMP2RANGE_TTEM,
        comp2range_preq=_COMP2RANGE_PREQ_TTEM,
        n_prereq=n_prereq,
        cert_fun_name="ILM_sep2023",
        attributes=("depth", "thick", "doi"),
        reducers={"depth": "mean", "thick": "mean", "doi": "mean"},
        var0_fn=_ttem_var0,
        kernel_fn=lambda a: np.maximum(a["depth"], 75.0),
        mask_fn=lambda a: a["doi"] < a["depth"],
    )


def _ttem_var0(a):
    # var0 = (0.5*1.2*thick)^2 ; sentinel 10000 where depth < 2.
    var0 = (0.5 * 1.2 * a["thick"]) ** 2
    var0[a["depth"] < 2.0] = 10000.0
    return var0
