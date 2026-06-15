# -*- coding: utf-8 -*-
"""geosigma.preprocess — per-layer preprocessing.

masks (``get_layermasks``), point drawing (``drawpoints``), windowed range/sill
extraction via a pluggable ``RangeSillEstimator``, and clustering (SOM +
post-processing + cluster stats).

Scaffold only (Phase 0); populated in **Phase 5** (depends on Phases 2 & 4).
"""
