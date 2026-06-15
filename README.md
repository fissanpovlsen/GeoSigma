# GeoSigma

GeoSigma is an open-source Python library for geostatistical simulation of 3D subsurface layer boundaries. Each layer boundary is modelled as a spatially variable Gaussian field, enabling users to generate multiple equally probable hydrostratigraphic realizations from a deterministic input model and quantify spatial uncertainty at every grid location.

The library is designed to be model-agnostic: the same workflow applies to any hydrostratigraphic model, independent of region or data source.

## Capabilities

- Construct spatial variance maps and correlation structures for layer boundaries from multiple geophysical data themes (TEM, PACES, gamma logs, seismic reflectors, etc.)
- Generate geostatistical realizations of 3D stratigraphic models via Sequential Gaussian Simulation
- Export results as GeoTIFF rasters for use in downstream modelling workflows
- Integrate with PEST for parameter perturbation and ensemble-based uncertainty propagation

## Installation

```bash
git clone https://github.com/rbm-geus/GeoSigma.git
cd GeoSigma
conda env create -f environment.yml
conda activate geosigma
pip install -e .
```

## Getting Started

The `examples/` directory contains standalone scripts demonstrating key workflows:

- `demo_geosigma_pipeline.py` — full workflow from surface input to geostatistical realizations
- `demo_synthetic_kriging_inversion.py` — synthetic kriging with masking
- `demo_inpox_basic_jutland_hydrostrat.py` — INPOX conditioning point selection on a real model

Example data (the Jutland hydrostratigraphic model) is included in `examples/data/`.

## Background

GeoSigma is developed at the Geological Survey of Denmark and Greenland (GEUS) as a clean Python reimplementation of a MATLAB-based workflow originally developed for generating stochastic realizations of the national-scale hydrostratigraphic model of Denmark. The original MATLAB source is included in `matlab_reference/` for reference and is not intended to be run directly.

Some core functions are adapted from the open-source [mGstat](https://github.com/cultpenguin/mGstat) MATLAB library (H. Madsen); those files carry individual credit notices.

## License

GNU General Public License v3.0 — see `LICENSE`.
