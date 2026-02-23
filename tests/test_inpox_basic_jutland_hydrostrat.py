# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 16:57:20 2026

@author: rbm
"""

import matplotlib.pyplot as plt
from utils import load_geotiff_grid, grid_coordinates_from_esri_meta
from geosigma import draw_points_inpox,ftot_laplace,laplacian_2d

z, meta = load_geotiff_grid("tests/data/Jutland_hydrostrat_model/top_surface.tif")
x, y, xx, yy = grid_coordinates_from_esri_meta(meta)

ext_vals = {
    "p0": 0.1,
    "a": 0.1,
    "b": 0.5,
    "c": 0.05,
    "d": 0.005,
    "e": 0.3,
    "g": 10,
    "x0": meta["xllcorner"],
    "y0": meta["yllcorner"],
}

points, plap, lapl = draw_points_inpox(z, ext_vals, dx=meta["cellsize"])

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Laplacian")
plt.imshow(lapl)

plt.subplot(1,3,2)
plt.title("Probability map")
plt.imshow(plap)

plt.subplot(1,3,3)
plt.title("Sampled points")
plt.imshow(points)

plt.show()

"it would be neat to  also in the test "show" the chosen transfer-function
" by providing a vector with laplacians linspace from say 0 to 20 where the ftot_laplace"
" could be computed and then visualized as a function of laplacian. Furthermore if this could be superimposed on a histogram of the laplacians for the surface "