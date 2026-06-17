# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Named certainty-decay kernels for the variance-map themes.

A *certainty function* maps the distance from a grid cell to the nearest
contributing data point into a certainty value (the reciprocal of variance).
All theme builders share these kernels; new named variants belong here and
nowhere else (see ``CLAUDE.md`` — *What to Avoid*).

Each kernel has the signature ``cert_fun(dist, range_, width, sill)``:

* ``dist``   distance from cell to nearest data point.
* ``range_`` effective correlation range (controls the Gaussian decay).
* ``width``  plateau half-width: inside it the kernel does not decay.
* ``sill``   certainty at the plateau (``1 / var0``).

All arguments broadcast as numpy arrays, so a kernel evaluates a whole grid
stack at once. The formulae are ported verbatim from the MATLAB
``certainty_function_definer.m`` / the root ``certainty_function_definer.py``;
the statistical model is preserved exactly.
"""

from __future__ import annotations

import numpy as np


def _frafa_apr2023(dist, range_, width, sill):
    """Flat plateau of height ``sill`` inside ``width``, Gaussian decay outside."""
    return np.where(
        dist <= width,
        sill,
        sill * np.exp(-3.0 * (dist - width) ** 2 / range_**2),
    )


def _ilm_sep2023(dist, range_, width, sill):
    """Gaussian decay from the plateau edge inside ``width``; zero outside."""
    return np.where(
        dist <= width,
        sill * np.exp(-3.0 * (dist - width) ** 2 / range_**2),
        0.0,
    )


def _ilm_oct2023(dist, range_, width, sill):
    """Gaussian decay measured from the cell (not the plateau edge); zero outside."""
    return np.where(
        dist <= width,
        sill * np.exp(-3.0 * dist**2 / range_**2),
        0.0,
    )


def _rbm_oct2023(dist, range_, width, sill):
    """Continuous two-piece Gaussian: damped decay inside, matched decay outside."""

    def _inside(d, r, w, s, damp):
        return s * np.exp(-damp * d**2 / r**2)

    def _outside(d, r, w, s):
        return s * np.exp(-3.0 * d**2 / r**2)

    sill_adjusted = sill + (sill - _inside(width, range_, width, sill, 1.5))
    return np.where(
        dist <= width,
        _inside(dist, range_, width, sill, 1.5),
        _outside(dist, range_, width, sill_adjusted),
    )


#: Registry of named certainty kernels.
_CERT_FUNCTIONS = {
    "FRAFA_apr2023": _frafa_apr2023,
    "ILM_sep2023": _ilm_sep2023,
    "ILM_oct2023": _ilm_oct2023,
    "RBM_oct2023": _rbm_oct2023,
}


def certainty_function(name: str = "FRAFA_apr2023"):
    """Return the named certainty kernel ``cert_fun(dist, range_, width, sill)``.

    Parameters
    ----------
    name : str
        One of :data:`available_certainty_functions`. Identified by a
        ``name_MONYYYY`` key, matching the MATLAB convention.

    Raises
    ------
    ValueError
        If ``name`` is not a registered kernel.
    """
    try:
        return _CERT_FUNCTIONS[name]
    except KeyError:
        raise ValueError(
            f"Unknown certainty function choice: {name!r}. "
            f"Available: {sorted(_CERT_FUNCTIONS)}"
        )


def available_certainty_functions():
    """Return the sorted names of all registered certainty kernels."""
    return sorted(_CERT_FUNCTIONS)
