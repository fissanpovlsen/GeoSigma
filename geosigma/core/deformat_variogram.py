import re
#import numpy as np

def deformat_variogram(txt):
    """
    Parse MATLAB-style variogram string into Python dict.
    This code is based upon a translation of mGstat, a Matlab geostatistical code library by Thomas Mejer Hansen  


    Parameters
    ----------
    txt : str
        Variogram string, e.g. '1 Sph(5)'

    Returns
    -------
    V_list : list of dict
        List of variogram dictionaries with keys:
        - 'type': model type
        - 'par1': sill
        - 'par2': range
    """
    V_list = []
    parts = re.findall(r"([\d\.eE\+\-]+)\s+([A-Za-z]+)\(([^\)]+)\)", txt)
    for par1, vtype, args in parts:
        args = [float(a) for a in args.split(",")]
        par2 = args[0]
        if len(args) > 1:
            print(
                "Warning: additional variogram parameters (anisotropy/angle) are ignored. Only isotropic is supported."
            )
        V_list.append({"type": vtype, "par1": float(par1), "par2": par2})
    return V_list
