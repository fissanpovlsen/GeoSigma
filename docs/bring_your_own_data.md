# Bring your own data

GeoSigma's variance-map themes are model-agnostic: the engine never touches raw
data. It sees only the **locations** where a source informs a layer boundary and
the per-location **attributes** that govern the uncertainty there, and from those
plus the distance from each grid cell to the nearest informed location it computes
the **variance of interpreting that boundary given this one type of information**.
Everything specific to a data type — a geophysical survey, a borehole, a
digitised interpretation — is expressed as a `ThemeSpec` and a small amount of
per-point preparation.

This document is the practitioner's guide to configuring a theme for your own
region and data. It is organised around the **distinct scientific choices** a
theme author must make. Each is a question that different geology or a different
data type could reasonably answer differently; none is an implementation detail.
There are twelve, in two groups:

- **Part A — choices expressed in the `ThemeSpec`** (eight). The spec is a plain
  dataclass; these are its fields and the formulae it carries.
- **Part B — choices in the per-point preparation layer** (four). These are made
  *before* the spec is applied, when a raw source is turned into the point cloud
  and attributes the engine consumes.

For each choice you get: the **question** in plain terms, the **parameter(s)**
that express the answer, **how to reason about it** for your own setting, and the
**Danish answers** as worked illustrations. The Danish values come from the
national hydrostratigraphic model (the DK-model); their scientific justification
is the subject of the accompanying Hydrogeology Journal article — cited below as
**[HJ citation]** (placeholder, to be filled on publication).

> **File formats are covered elsewhere.** How the point data, manifest, and
> complexity grid are laid out on disk — column conventions, the `attr__LNN`
> per-layer blocks, the complexity contract — is specified in
> [`theme_format.md`](theme_format.md). This document is about *what to put in
> them and why*, not their syntax; it cross-references rather than restates.

---

# Part A — choices expressed in the `ThemeSpec`

A `ThemeSpec` is the uncertainty model for one information source. Its fields are
the answers to choices 1–8. Two of its callables — `var0_fn` (*how certain is the
interpretation where the source sees the boundary?*) and `mask_fn` (*can the
source see the boundary at all?*) — are the model's two central knobs; the rest
tune the spatial behaviour around them.

> ### A design invariant: you do not weight themes
>
> A natural first question is *"how do I weight my themes against each other?"*
> You don't — and this is deliberate, not a missing feature. Themes combine as
> **Gaussian precision addition**: the final precision (1/variance) at a cell is
> the sum of the themes' precisions, `1 / Σ(1/θ)`. That sum is the correct
> posterior *only if each theme contributes independent information*. Introducing
> combination weights would assert that some evidence counts for more or less than
> its own stated precision — which breaks the independence the addition relies on
> and no longer corresponds to any posterior.
>
> A method's relative importance is therefore expressed **inside the theme, through
> the uncertainty it claims** — a smaller `var0`, a wider certainty plateau, a
> deeper reach all make a method count for more where it applies, exactly as much
> as its own physics warrants. If you find yourself wanting to down-weight a
> theme, the correct lever is to raise its claimed variance (choice 6) or tighten
> its reach (choice 7), not to add a weight. Get each theme's self-described
> uncertainty right and the combination takes care of itself.

## Choice 1 — Spatial support of a data point (search window)

**Question.** How far from a data point may its measurement be allowed to inform a
grid cell?

**Parameter.** `ThemeSpec.search_radius` — the half-width, in grid cells, of the
square window slid over the grid. Within that window the single nearest informed
location is found; beyond it a point contributes nothing.

**Reasoning for your setting.** This is a statement about **data density and the
support of a single measurement**, expressed in cells, so it depends on your grid
resolution. A dense, areally-averaging survey justifies a wider window than sparse
point measurements; a method whose footprint is essentially a point argues for a
small radius. Set it large enough that the nearest-point search rarely comes up
empty in areas you consider covered, but not so large that a measurement is
allowed to speak for ground it never sampled. (The correlation *range*, choice 3,
governs how fast certainty then decays with distance; `search_radius` only bounds
how far the search looks.)

**Danish illustration [HJ citation].** PACES `search_radius = 1`; reflection
seismic `search_radius = 8`; the TEM methods and the borehole logs use `6`.

## Choice 2 — Combining coincident points (the reducer)

**Question.** When several data points are the nearest to a cell, or tie at exactly
the minimum distance, how do you reduce their per-layer attribute values to the
single value the cell will use — average them, take the most conservative, take
the first?

**Parameter.** `ThemeSpec.reducers` — a mapping from each attribute name to one of
`"mean"`, `"min"`, `"max"`, or `"first"`. `"first"` is special: it takes the value
from the first point (lowest index) whose *window* contains the cell, not the
nearest — used when an attribute is a categorical label rather than a measurement.

**Reasoning for your setting.** Choose per attribute according to what the tie
*means*. Averaging suits a continuous measurement with symmetric noise. A
conservative `min`/`max` suits an attribute where the safe estimate is the
smallest/largest — e.g. taking the *thinnest* reported layer where a thin layer
means less certainty. `"first"` suits a per-acquisition label (an instrument or
survey type) where averaging is meaningless. If your points never coincide, the
reducer rarely bites — but set it deliberately, because dense data will exercise
it.

**Danish illustration [HJ citation].** Interpreted depth is always reduced by
`mean`. Layer thickness is reduced by `min` for the logs and SkyTEM, `mean` for
many-layer and towed TEM, and `max` for reflection seismic. MEP's acquisition
type uses `first`.

## Choice 3 — Correlation range vs geological complexity (and a deep-layer regime)

**Question.** How far does a boundary interpretation stay spatially correlated —
i.e. how quickly does certainty decay with distance — as a function of the *local
geological complexity*? And do deep layers follow a different rule from shallow
ones?

**Parameter.** `ThemeSpec.range_model` — a list of `RangeGroup(layers,
range_by_complexity)`. Each group covers a set of layers and gives the correlation
range for each complexity class (index 0 is the unused "unknown" slot; indices 1…K
are complexity classes low→high). A single group with `layers="all"` means one
regime for every layer; two groups split the layer axis into a shallow and a deep
regime. Complexity classes come from an optional class grid (see the complexity
contract in [`theme_format.md`](theme_format.md)); with no grid the engine assumes
uniform class 1.

**Reasoning for your setting.** Two decisions live here. First, **does correlation
length depend on geological complexity in your area, and how?** Where structure is
simple and layer-cake, interpretations stay correlated over long distances (a long
range); where geology is complex, they decorrelate quickly (a short range). If you
have a complexity classification, encode that dependence as the per-class table; if
you do not, use a single uniform range and declare `complexity: none`. Second,
**do deeper layers behave differently?** If a distinct deep stratigraphy (in
Denmark, the pre-Quaternary) has its own correlation behaviour, split the layers
into two `RangeGroup`s. The split index itself is a property of your model, not of
the theme.

**Danish illustration [HJ citation].** Most themes use ranges of 500 / 400 / 250 /
100 m for complexity classes 1–4, with a pre-Quaternary override of 500 m for the
deep layers. Towed TEM instead uses a flat 100 m for every class and depth.

## Choice 4 — The certainty decay kernel

**Question.** Once you are beyond the near-source plateau, *how* does certainty
fall off with distance — the shape of the decay curve?

**Parameter.** `ThemeSpec.cert_fun_name` — the **name** of a kernel in GeoSigma's
registry. A kernel maps `(dist, range_, width, sill)` to a certainty value; the
spec stores only the name so it stays serialisable to and from YAML.

A *certainty function* (kernel) is the decay model: given the distance from a cell
to its nearest contributing point, the effective correlation range, the plateau
half-width (choice 5), and the plateau certainty `sill = 1/var0` (choice 6), it
returns the certainty at that cell. GeoSigma ships four named kernels:

- `FRAFA_apr2023`
- `ILM_sep2023`
- `ILM_oct2023`
- `RBM_oct2023`

These encode **specific scientific choices** made in producing the DK-model
variance maps — each a particular assumption about how a Danish data type's
certainty decays with distance. They are not generic defaults. **If you are
reproducing the Danish results, use the preset names as-is**; substituting a
different kernel will change the output.

**Reasoning for your setting.** If your data type or geological setting implies a
different decay assumption, register your own kernel and reference it by name — the
same mechanism the built-ins use. Register once at start-up:

```python
from geosigma.themes import register_certainty_function

def linear_falloff(dist, range_, width, sill):
    """Certainty falls linearly from `sill` at the plateau to zero at `range_`."""
    import numpy as np
    return sill * np.clip(1.0 - dist / range_, 0.0, None)

register_certainty_function("MYORG_jan2026", linear_falloff)  # NAME_MONYYYY
```

A kernel must declare exactly these four positional parameters, in order:

| argument | meaning |
|----------|---------|
| `dist`   | distance from the grid cell to its nearest contributing data point |
| `range_` | effective correlation range (the decay length; from choice 3) |
| `width`  | plateau half-width — inside it the kernel does not decay (choice 5) |
| `sill`   | certainty at the plateau, `1 / var0` (choice 6) |

All arguments broadcast as numpy arrays, so a kernel evaluates a whole grid stack
at once. `register_certainty_function` validates the signature and raises
`TypeError` if it does not match, or `ValueError` if the name is already
registered (pass `overwrite=True` to replace deliberately). Then reference it from
the spec exactly like a built-in: `cert_fun_name="MYORG_jan2026"`. New library
kernels belong in `geosigma/themes/certainty_functions.py`, never scattered across
theme scripts.

**Danish illustration [HJ citation].** PACES and reflection seismic use
`FRAFA_apr2023` (a flat plateau, Gaussian decay outside); the TEM and log themes
use `ILM_sep2023` (Gaussian decay measured from the plateau edge).

## Choice 5 — The near-source plateau half-width

**Question.** Out to what distance is the source's certainty at full strength
before it begins to decay — and does that distance grow with the boundary's depth?

**Parameter.** `ThemeSpec.kernel_fn` — a callable on the reduced attribute grids
returning the plateau half-width `width` (the `width` argument the kernel receives).
It may return a constant or a value that varies per cell.

**Reasoning for your setting.** The plateau is the neighbourhood in which you treat
the source as *directly* informative rather than extrapolated. A method that
genuinely averages over an area supports a wide plateau; a near-point method a
narrow one. Whether the plateau should **grow with depth** is a physical question:
if a method's lateral footprint widens with depth (as many soundings' do), a
depth-dependent width captures that; if not, a constant is right. Return a constant
for depth-independent support, or a function of `depth` for a footprint that
widens.

**Danish illustration [HJ citation].** The logs use a constant 150; PACES/PACEP/MEP
a constant 75; reflection seismic a constant 1 (essentially no plateau). SkyTEM
uses `max(2·depth, 75)` and the other TEM methods `max(depth, 75)` — a plateau that
widens with depth.

## Choice 6 — The base interpretation variance (`var0`)

**Question.** Where the source *does* see the boundary, how uncertain is the
interpreted depth — and does that uncertainty stay constant, or grow with depth,
layer thickness, or acquisition mode?

**Parameter.** `ThemeSpec.var0_fn` — a callable on the reduced attribute grids
returning `var0` (the variance at the plateau; the kernel uses `sill = 1/var0`).
This is the magnitude of the source's claimed uncertainty and, per the design
invariant above, is where a method's relative importance is expressed.

**Reasoning for your setting.** This is the heart of the theme. There are two broad
archetypes. Some methods have sensitivity roughly **constant with depth** — a flat
`var0` (e.g. reflection picks whose quality does not degrade downward). Others
**degrade with depth** — `var0` grows as the boundary deepens (most electrical and
electromagnetic methods). Write the formula that your understanding of the method's
physics dictates; it may also depend on a layer's thickness (a thicker layer
interpreted more confidently) or on an acquisition mode. Calibrate its magnitude to
a real, defensible uncertainty in the units of the boundary (metres² of depth
variance) — because this number, not a weight, is what sets how much the method
counts in the combination.

**Danish illustration [HJ citation].** Reflection seismic uses a constant
`(0.5·15)² = 56.25`; the gamma log a constant `2.25`; PACES a depth-growing
`(0.5·max(2, 0.25·depth))²`; the thickness-driven TEM methods `(0.5·1.2·thick)²`;
MEP switches formula by acquisition type (Wenner-2D vs other).

## Choice 7 — Where the source can see the boundary (reach and shallow floor)

**Question.** Under what conditions does this source carry *no* usable information
about a boundary — because the boundary lies deeper than the method reached, or
shallower than it can resolve?

**Parameter.** `ThemeSpec.mask_fn` — a callable returning a boolean grid of cells
to set to the no-information sentinel (the deep/reach cutoff). The **shallow
cutoff** is conceptually part of this same choice — the minimum depth below which
the source is untrustworthy — but in the current code it is implemented as a
sentinel *inside* `var0_fn` rather than in `mask_fn`. *(Implementation note: this
split is historical; unifying the shallow floor into `mask_fn` is a possible
v0.2.0 polish, deferred to avoid numeric changes near release.)*

**Reasoning for your setting.** Every source has a domain of validity. The **deep
limit** is usually a depth-of-investigation or penetration: below it the signal no
longer constrains the boundary, so the cell must be no-information rather than
confidently wrong — mask where `reach < depth`. The **shallow limit** is where a
method loses resolution near the surface (a first-arrival blind zone, a minimum
layer thickness): below it, likewise, no information. State both honestly; masking
too little is worse than masking too much, because an overconfident wrong value
propagates into the combination as if it were evidence.

**Danish illustration [HJ citation].** The TEM methods mask where the
depth-of-investigation is shallower than the boundary (`doi < depth`); the logs
mask where the logged thickness is less than the boundary depth; reflection seismic
masks layers thinner than 100 m. Shallow floors vary by method — e.g. SkyTEM
excludes boundaries above 5 m, the resistivity log above 10 m.

## Choice 8 — Which boundaries the source constrains at all (active layers)

**Question.** Does this source inform every modelled boundary, or only a specific
subset of them?

**Parameter.** `ThemeSpec.active_layers` — `None` (the default) means the source is
applied to every layer; otherwise an explicit set of layer indices, and all other
layers receive the no-information sentinel. When the set is derived from the data
it is usually carried as a manifest `spec_params` value (see
[`theme_format.md`](theme_format.md)).

**Reasoning for your setting.** Some methods, by their nature, only bear on certain
horizons — a method that images a specific reflector, a log run only over part of
the section. If yours is one, restrict it to those layers so it does not silently
claim (even as no-information) to speak about boundaries it never addressed. If the
subset is a property of *each dataset* rather than the method in general, compute it
during preparation and pass it through `spec_params`. Most themes leave this
`None`.

**Danish illustration [HJ citation].** Reflection seismic is restricted to the
layers whose geophysical thickness exceeds 1000 m (a data-derived set); every other
theme applies to all modelled layers.

---

# Part B — choices in the per-point preparation layer

The preparation layer turns a raw source into the point cloud and attribute arrays
the engine consumes. It runs *before* the spec, in an **adapter** you write for
your data (see [`theme_format.md`](theme_format.md) for the file the adapter
produces). The four choices below are made here, not in the `ThemeSpec` — but they
are scientific choices all the same, and should be recorded in the bundle's
provenance notes.

## Choice 9 — Filling missing depth-of-investigation (and other attribute gaps)

**Question.** When an observation does not report an attribute the theme needs —
most commonly a sounding with no recorded depth-of-investigation — how do you
estimate the missing value?

**Parameter(s).** Two mechanisms, and *which one to use turns on whether the
estimate depends on the observation's location*:

- **Location-dependent → resolve it per-sounding, before windowing**, with
  `geosigma.preprocess.doi.resolve_doi` (`horizon_top`, `horizon_thickness`,
  `min_horizon_thickness`, `fallback_doi`, `fallback_when_empty`). Its estimate
  uses a *conductive horizon* sampled at each sounding's own location, so it cannot
  be deferred: once soundings are reduced to per-cell values, the per-sounding
  horizon depth is gone.
- **Location-independent → fill per-cell, after windowing**, in the spec's
  `ThemeSpec.post_reduce_fn`. A constant or a global mean does not depend on where
  a sounding sits, so it can (and should) be applied downstream, after the
  nearest-point reduction.

**Reasoning for your setting.** First ask whether your fill depends on location. A
physically-motivated estimate — e.g. capping the depth-of-investigation at a
conductive layer that absorbs the signal — is location-dependent and belongs in the
pre-window step. A blanket "missing → 15 m" or "missing → mean of what we have" is
location-independent and belongs in `post_reduce_fn`. Putting a location-dependent
fill downstream silently corrupts it; putting a constant fill upstream just makes
extra work. If nothing is missing, you need neither.

**Danish illustration [HJ citation].** The airborne and few/many/towed TEM methods
estimate a missing depth-of-investigation from the Palaeogene conductive clay,
capping it at the clay top where the clay is at least ~10 m thick and otherwise
falling back to the mean recorded depth-of-investigation (150 m if none is
recorded) — all per-sounding, before windowing. PACEP instead fills a missing value
with a constant 15 m, and MEP with the mean of its recorded values — both per-cell,
after windowing.

## Choice 10 — Temporal currency filter

**Question.** Which observations are recent or relevant enough to include, rather
than superseded by later data or by the model epoch?

**Parameter.** A date filter in the adapter, comparing each observation's survey
year against the model year. The rule (and the column it reads) is recorded in the
bundle's `_notes` (see [`theme_format.md`](theme_format.md)).

**Reasoning for your setting.** Decide what "current" means for your compilation:
keep observations up to the model epoch, drop pre-standardisation vintages, or
whatever your data quality demands. The point is that the cut is an explicit,
recorded decision — a reader must be able to reproduce your row set — not an
incidental side effect of which files you happened to load.

**Danish illustration [HJ citation].** Points are kept where the survey year does
not exceed the model year; reflection seismic and the borehole logs read the survey
year from a `d_year` column, the geophysical soundings from `m_year`.

## Choice 11 — De-duplicating coincident observations

**Question.** When several records share the same location, which one survives into
the point cloud?

**Parameter.** A de-duplication rule in the adapter, keyed on coordinates. The rule
is recorded in `_notes`.

**Reasoning for your setting.** Coincident records — re-processings, re-surveys,
duplicated exports — must be collapsed, and the rule is a real choice: keep the
most recent, keep the first, or average. Whatever you choose, apply it consistently
and record it, because it changes which measurement speaks for that location.

**Danish illustration [HJ citation].** Records are de-duplicated on their `(x, y)`
coordinates, keeping the last occurrence (matching the reference pipeline's
unique-by-rows behaviour). The rule is the same for all ten themes.

## Choice 12 — Mapping source columns to attributes (and fixed-value overrides)

**Question.** Which raw column supplies each attribute the spec consumes — the
interpreted depth, a thickness, a depth-of-investigation, an acquisition type — and
is any attribute deliberately set to a fixed value rather than read from the data?

**Parameter.** The adapter's mapping from source columns into the `ThemeData`
attribute blocks (whose on-disk form is specified in
[`theme_format.md`](theme_format.md)), plus any constant overrides it writes.

**Reasoning for your setting.** This is where your data's idiosyncrasies are
absorbed so the spec can stay generic: pick the column that means "interpreted
boundary depth", the one that means "how deep the method saw", and so on, and name
them as the attributes the spec expects. Sometimes the right answer is a **constant,
not a column** — when a method's reach is effectively fixed, encoding it as a single
value is more honest than a spurious per-point column. Record which columns fed
which attributes (and whether they are raw or already processed) in the provenance
notes.

**Danish illustration [HJ citation].** Interpreted depths and thicknesses come from
the model's per-layer depth and thickness arrays; the depth-of-investigation from a
`doilower` column; MEP's acquisition type from a `datasubtype` column (1 if it
begins "wen", else 0). The two borehole-log themes read UTM coordinates from
differently-named columns than the geophysical themes. PACES supplies only depth and
fixes its reach at a constant 18 m (expressed through the mask, choice 7) rather
than reading a per-sounding value.

---

# A complete worked example

To show the whole path start to finish, here is a **fictional, non-Danish** theme:
`refraction_bedrock`, a seismic-refraction survey that picks the depth to bedrock in
an alluvial basin. Each sounding reports an interpreted bedrock depth and a
`spread_reach` — the maximum depth the spread length can resolve, its
depth-of-investigation analogue. Bedrock is a single deepening surface, so we treat
every modelled layer the same and use no complexity grid.

We make each of the twelve choices explicitly.

```python
import numpy as np
from geosigma.themes import (
    ThemeSpec,
    RangeGroup,
    register_certainty_function,
    build_theme,
)

# --- Choice 4: certainty kernel -------------------------------------------
# Refraction certainty is high near a shot and falls off smoothly; we register
# a simple Gaussian-from-the-edge kernel under a NAME_MONYYYY name.
def acme_refraction_jan2026(dist, range_, width, sill):
    inside = dist <= width
    return inside * sill * np.exp(-3.0 * (dist - width) ** 2 / range_**2)

register_certainty_function("ACME_refraction_jan2026", acme_refraction_jan2026)

# --- Choices 3, 5, 6, 7 as closures on the reduced attribute grids ---------
def var0(attrs):                       # Choice 6: uncertainty grows with depth
    depth = attrs["depth"]
    v = (0.5 * np.maximum(2.0, 0.20 * depth)) ** 2
    v[depth < 3.0] = 100000.0          # Choice 7 (shallow floor): blind zone < 3 m
    return v

def width(attrs):                      # Choice 5: constant 60 m plateau
    return np.full_like(attrs["depth"], 60.0)

def reach_mask(attrs):                 # Choice 7 (deep reach): below the spread DOI
    return attrs["spread_reach"] < attrs["depth"]

spec = ThemeSpec(
    name="refraction_bedrock",
    search_radius=3,                   # Choice 1: 3-cell support
    range_model=[                      # Choice 3: one regime, uniform 700 m range
        RangeGroup(layers="all", range_by_complexity=[0.0, 700.0]),
    ],                                 #   (index 0 unused; class 1 = 700 m)
    cert_fun_name="ACME_refraction_jan2026",   # Choice 4
    attributes=("depth", "spread_reach"),
    reducers={"depth": "mean", "spread_reach": "mean"},   # Choice 2
    var0_fn=var0,                      # Choice 6
    kernel_fn=width,                   # Choice 5
    mask_fn=reach_mask,                # Choice 7 (deep reach)
    active_layers=None,                # Choice 8: informs every layer
)
```

With `complexity: none` declared in the manifest (choice 3's "no complexity grid"),
the theme is built by handing the spec a `ThemeData` and the working grid:

```python
grid = build_theme(data, spec, grid_x, grid_y, n_layers)
```

The **preparation-layer choices (9–12)** are made in the adapter that produces
`data` — writing the three-file bundle specified in
[`theme_format.md`](theme_format.md):

- **Choice 9 (attribute gaps).** `spread_reach` is measured for every sounding, so
  nothing needs filling. Were some missing and estimable from a known signal-limiting
  horizon, we would call `resolve_doi` before windowing; a blanket constant fill
  would instead go in the spec's `post_reduce_fn`.
- **Choice 10 (currency).** Keep soundings whose `campaign_year` does not exceed the
  model year.
- **Choice 11 (dedup).** Where re-shot lines coincide, keep the last occurrence at
  each `(x, y)`.
- **Choice 12 (column mapping).** Map the survey's `easting`/`northing` to the
  coordinate columns, its per-layer `bedrock_depth_*` picks to the `depth` block,
  and its `max_resolved_depth` to `spread_reach`; no fixed overrides.

That is a complete theme: eight spec choices in Python, four preparation choices in
the adapter, and a variance map out of `build_theme` — with no weighting anywhere,
because each theme's honesty about its own uncertainty is what the combination
needs (see the design invariant in Part A).
