# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:05:30 2026

@author: rbm
"""

import numpy as np
import rasterio


def load_geotiff_grid(fname):
    """
    Load a GeoTIFF raster.

    Returns
    -------
    z : 2D ndarray
        Raster values
    meta : dict
        Dictionary with grid metadata compatible with ESRI ASCII style
    """
    with rasterio.open(fname) as ds:
        z = ds.read(1)  # first band
        transform = ds.transform

        meta = {
            "ncols": ds.width,
            "nrows": ds.height,
            "cellsize": transform.a,      # pixel width
            "xllcorner": transform.c,
            "yllcorner": transform.f - ds.height * transform.e,
            "nodata_value": ds.nodata,
            "crs": ds.crs,
            "transform": transform,
        }

    return z, meta