\# MATLAB Reference Code



\## Purpose



This folder contains the original MATLAB code that forms the basis for the GeoSigma Python package. The code was developed as part of a project to generate hydrostratigraphic realizations of the national-scale hydrostratigraphic model of Denmark using geostatistical simulation.



The MATLAB code is included here strictly as reference material for developers working on GeoSigma. It is not intended to be run standalone, as it contains hardcoded data paths and dependencies that are specific to the original project environment.



\---



\## What the Code Does



The MATLAB code implements a geostatistical simulation workflow for generating multiple equally probable hydrostratigraphic realizations from a deterministic hydrostratigraphic model. The key steps in the workflow include



\- Reading and preprocessing hydrostratigraphic model input data

\- Performing geostatistical simulation (e.g. Sequential Gaussian Simulation or similar)

\- Generating multiple stochastic realizations of hydrostratigraphic layer geometries

\- Post-processing and exporting results



\---



\## Why It Cannot Be Run Directly



The original MATLAB code was written for a specific project context and contains



\- Hardcoded file paths pointing to local or network data storage

\- Project-specific data files (not included in this repository due to size)

\- Dependencies on local MATLAB toolboxes and helper functions

\- Denmark-specific model inputs that are not generalised



\---



\## Relationship to GeoSigma



GeoSigma is a clean, general-purpose Python reimplementation of the principles in this MATLAB code. The goals of the translation are to



\- Remove all hardcoded paths and project-specific references

\- Make the workflow applicable to any hydrostratigraphic model, not just the Danish one

\- Produce a well-documented, installable Python package

\- Follow modern Python packaging and software development standards



The table below maps the original MATLAB files to their Python equivalents in GeoSigma (update as translation progresses)



&#x20;MATLAB File  Description  GeoSigma Equivalent 

\---------

&#x20;(to be filled in)   



\---



\## Data Files



The data files required to run the original MATLAB code are not included in this repository due to their size. They reside in a separate data directory outside of version control.



\---



\## Contact



For questions about the original MATLAB code or the GeoSigma translation project, please open an issue in the GeoSigma repository.

