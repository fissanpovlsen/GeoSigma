import numpy as np
from ReadSurfer7 import ReadSurfer7
def import_complexitymap(X, Y, reg):
    print("Loading Geological Complexity Map")

    # Placeholder for reading Surfer7 grid file
    # Replace this with actual implementation of ReadSurfer7
    M, info = ReadSurfer7('data\Kompleksitetskort.grd')  # You need to define or import this

    # Padding
    Nrow, Ncol = M.shape
    PadSize = 400
    BigM = np.zeros((Nrow + 2 * PadSize, Ncol + 2 * PadSize))
    BigM[PadSize:-PadSize, PadSize:-PadSize] = M
    BigM[BigM < -1] = 0
    M = BigM

    # Adjust coordinates based on region
    if reg in ['Jylland', 'Fyn', 'Sjælland', 'AnholtLæsø']:
        Xg = info['UTM_X'] + 50
        Yg = info['UTM_Y'] + 50
    elif reg == 'Fyn-MST':
        Xg = info['UTM_X']
        Yg = info['UTM_Y']
    else:
        raise ValueError("Unknown region")

    # Extend coordinate grids to match padded matrix
    Xg = np.concatenate([
        np.array([int((i) * 100) + Xg[0] for i in range(-PadSize, 0)]),
        Xg,
        np.array([int((i) * 100) + Xg[-1] for i in range(1, PadSize + 1)])
    ])
    Yg = np.concatenate([
        np.array([int((i) * 100) + Yg[0] for i in range(-PadSize, 0)]),
        Yg,
        np.array([int((i) * 100) + Yg[-1] for i in range(1, PadSize + 1)])
    ])

    # Find matching indices
    IX1 = [i for i, val in enumerate(Xg) if val in X]
    IY1 = [i for i, val in enumerate(Yg) if val in Y]

    # Extract map
    map_out = M[np.ix_(IY1, IX1)]
    return map_out