
import numpy as np
from .matlab_helpers import matlab_meshgrid

def grid_coordinates_from_esri_meta(meta):
    nx = meta["ncols"]
    ny = meta["nrows"]
    dx = meta["cellsize"]

    x0 = meta["xllcorner"]
    y0 = meta["yllcorner"]

    x = x0 + dx * (np.arange(nx) + 0.5)
    y = y0 + dx * (np.arange(ny) + 0.5)

    xx, yy = matlab_meshgrid(x, y)

    return x, y, xx, yy
