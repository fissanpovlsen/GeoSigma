import numpy as np

def load_esri_ascii_grid(fname):
    """
    Load an ESRI ASCII grid (.asc).

    Returns
    -------
    data : 2D ndarray (ny, nx)
    meta : dict with keys:
        ncols, nrows, xllcorner, yllcorner, cellsize, nodata_value
    """
    meta = {}
    with open(fname, "r") as f:
        # Read header (first 6 lines)
        for _ in range(6):
            key, value = f.readline().split()
            meta[key.lower()] = float(value) if "." in value else int(value)

        # Read grid values
        data = np.loadtxt(f)

    # Replace nodata with NaN (much safer downstream)
    nodata = meta.get("nodata_value", None)
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    return data, meta
