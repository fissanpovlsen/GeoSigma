# GeoSigma

GeoSigma – an open-source Python library for geostatistical uncertainty modeling of 3D subsurface layer boundaries.
Each layer is represented as a spatially variable Gaussian field, allowing users to ascribe and analyze uncertainty at every grid location.

The library includes tools for:

Constructing spatial variance maps and correlation structures for layer boundaries

Sampling geostatistical realizations of 3D stratigraphic models (optional)

Visualizing and exporting results for further analysis

GeoSigma can thus be used both for generating full probabilistic model realizations and for standalone calculation of uncertainty maps, giving flexibility depending on your workflow.



\# Setting up GeoSigma

1\. Clone the repository:

&nbsp;  git clone https://github.com/your-org/GeoSigma.git

2\. Create the Conda environment:

&nbsp;  conda env create -f environment.yml

3\. Activate the environment:

&nbsp;  conda activate geosigma

