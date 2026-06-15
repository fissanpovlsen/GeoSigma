# -*- coding: utf-8 -*-
"""geosigma.themes — STANDALONE variance-map tool.

Builds per-layer variance grids from geophysical data sources (today: the
root-level ``get_*_theme.py`` scripts) onto a shared cert-decay kernel, then
combines them (``1/sum(1/themes)``, minimum-map floor, corr/tie maps).

Runnable on its own (``python -m geosigma.themes ...``) and **independent of
kriging**: the pipeline consumes a variance-grid stack as an input, whether
produced here or supplied by the user.

Scaffold only (Phase 0); populated in **Phase 3**.
"""
