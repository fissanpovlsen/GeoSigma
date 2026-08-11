# Deferred decisions

Decisions that have been **made and deliberately postponed**, together with the
reasoning behind postponing them. The point of writing them down is that a
deferred decision is otherwise indistinguishable from an oversight: without the
rationale, the next person either rediscovers the whole argument or, worse, acts
on it at the wrong moment.

This is not a general backlog. The phased translation roadmap — which MATLAB
stages are ported and which are still to come — lives in `TRANSLATION_PLAN.md`.
Only decisions with a *deliberate* "not yet, and here is why" belong here.

---

## Packaging and dependencies

### Dependency upper caps — deferred

**State.** `pyproject.toml` declares lower bounds only (`numpy>=1.24`,
`pandas>=1.5`, `matplotlib>=3.6`, `rasterio>=1.3`, `affine>=2.3`, `pyyaml>=6`).
Nothing is capped, so a future major release of any dependency will be accepted
by the resolver.

**Recommendation on file, when the decision is taken.** Cap **only**
`rasterio<2` and `affine<4`. Both are small libraries whose major releases would
reshape the exact APIs this package calls, and neither is a hub dependency, so
capping them will not deadlock a user's resolver.

Leave **numpy, pandas and matplotlib uncapped**, per SPEC 0. Capping a hub
dependency blocks users from any environment where a co-installed package
requires the newer major — a concrete, immediate cost paid against a
hypothetical future break. The better guard is a scheduled CI job that installs
against unpinned latest: a red build gives early warning of a real
incompatibility, whereas a cap only converts a possible future failure into a
certain present one.

Supporting the loose bounds: the dependency API surface actually used is small
and long-stable — pandas is only `read_csv` and `.to_numpy`, rasterio only
`open` and `transform.from_origin`, affine only the `Affine` constructor, pyyaml
only `safe_load` / `safe_dump`, and nothing uses a numpy API removed in 2.0.

### `requirements-paper.txt` — write it at figure-generation time, not before

When the figures for the accompanying Hydrogeology Journal article are
generated, freeze that environment to a `requirements-paper.txt` of exact
versions and cite it in the article's data-availability statement.

**Why not now.** Version *ranges* do not make results reproducible; only exact
pins do. A lockfile written today would record an environment that did **not**
produce the published results — precisely the substitution that makes a
reproducibility claim untrue. Writing it at figure time is what makes the claim
literally correct, so the timing is the whole point of the decision.

### matplotlib as an optional extra — v0.2.0

`geosigma/inpox.py` imports `matplotlib.pyplot` at module level, so
`import geosigma` pulls the full plotting stack into any process that only wants
the geostatistical core. Deferring that import into
`visualize_transfer_function` — its only consumer — would let matplotlib move
out of the required dependencies and into an extra.

**Why not now.** It changes library import behaviour, which was out of scope for
v0.1.0.

---

## Library behaviour

### Shallow-floor unification into `mask_fn` — v0.2.0

A theme's **deep** cutoff (the source's reach) is expressed in
`ThemeSpec.mask_fn`, but its **shallow** cutoff — the minimum depth below which
the source is untrustworthy — is implemented as a sentinel written *inside*
`var0_fn`. The two are conceptually the same choice ("where can this source see
the boundary at all?"), and the split is historical rather than principled.
Unifying the shallow floor into `mask_fn` is the v0.2.0 polish.

**Why not now.** The two paths do not produce identical numbers, and the change
was held back to avoid numeric movement close to release.

This quirk is user-visible, so it is also documented where a theme author meets
it — choice 7 in [`bring_your_own_data.md`](bring_your_own_data.md). That note
stays: it explains behaviour someone writing a spec today has to work with.
Keep the two consistent if either is revised.
