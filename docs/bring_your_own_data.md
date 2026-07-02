# Bring your own data

GeoSigma's variance-map themes are model-agnostic: the engine only ever sees
plain coordinate and attribute arrays, so any information source — geophysical
survey, borehole, geological interpretation — can be expressed as a `ThemeSpec`
and fed through `build_theme`. This document collects the extension points for
running the themes on data other than the Danish DK-model reference set.

## Certainty functions

A *certainty function* (kernel) maps the distance from a grid cell to its nearest
contributing data point into a certainty value (the reciprocal of variance). It
is the decay model that controls how fast a source's information falls off with
distance.

### The four built-in kernels

GeoSigma ships four named kernels, registered under these names:

- `FRAFA_apr2023`
- `ILM_sep2023`
- `ILM_oct2023`
- `RBM_oct2023`

These reflect **specific scientific choices** made during production of the
variance maps for the Danish national hydrostratigraphic model (the DK-model).
They are not generic defaults — each encodes a particular assumption about how a
Danish geophysical data type's certainty decays with distance. The argued basis
for these choices will be documented in the accompanying Hydrogeology Journal
article (cite as **[HJ citation]** — placeholder, to be filled when the article
is published).
<!-- TODO: replace [HJ citation] with full reference once article is published -->

**If you are reproducing the Danish results, use the preset names as-is.** Do not
substitute your own kernel, or the output will no longer match the published
DK-model variance maps.

### Registering your own kernel

If your data type or geological setting implies a different decay assumption, you
can supply your own kernel and reference it by name from a `ThemeSpec` — the same
way the built-ins are referenced. Register it once at start-up:

```python
register_certainty_function(name, fn)
```

The spec stores only the **name string** (`cert_fun_name`), never the callable
itself; this keeps specs serialisable to/from YAML. Both built-in and
user-registered kernels are resolved by name at `build_theme` time.

#### Required signature

A kernel must declare exactly these four positional parameters, in order:

```python
def my_kernel(dist, range_, width, sill):
    ...
```

| argument | meaning |
|----------|---------|
| `dist`   | distance from the grid cell to its nearest contributing data point |
| `range_` | effective correlation range (controls the decay length) |
| `width`  | plateau half-width — inside it the kernel does not decay |
| `sill`   | certainty at the plateau (`1 / var0`) |

All arguments broadcast as numpy arrays, so a kernel evaluates an entire grid
stack at once. `register_certainty_function` validates the signature with
`inspect.signature` and raises `TypeError` if it does not match; it raises
`ValueError` if the name is already registered (pass `overwrite=True` to replace
deliberately).

#### Minimal example

```python
import numpy as np
from geosigma.themes import (
    register_certainty_function,
    ThemeSpec,
    build_theme,
)

def linear_falloff(dist, range_, width, sill):
    """Certainty falls linearly from `sill` at the plateau to zero at `range_`."""
    return sill * np.clip(1.0 - dist / range_, 0.0, None)

# Register once, under a name following the `NAME_MONYYYY` convention.
register_certainty_function("MYORG_jan2026", linear_falloff)

# Reference it by name from a spec — identical to using a built-in preset.
spec = ThemeSpec(
    name="my_source",
    search_radius=4,
    range_model=...,                 # see RangeGroup
    cert_fun_name="MYORG_jan2026",   # <- your registered kernel, by name
    attributes=("depth",),
    reducers={"depth": "mean"},
    var0_fn=...,
    kernel_fn=...,
    mask_fn=...,
)

grid = build_theme(data, spec, grid_x, grid_y, n_layers)
```

New kernels belong in `geosigma/themes/certainty_functions.py` (or registered at
runtime as above) — never scattered across theme scripts (see `CLAUDE.md`).
