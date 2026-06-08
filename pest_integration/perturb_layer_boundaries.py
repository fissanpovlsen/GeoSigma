# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:30:02 2026

@author: RBM, GEUS
"""

import os
import glob
from pathlib import Path

import rasterio
import numpy as np


def perturb_layer_boundaries(
    folder,
    k=1,
    method="shift",
    output_folder="perturbed",
):
    """
    Perturb layer-boundary GeoTIFFs.

    Parameters
    ----------
    folder : str or Path
        Folder containing the base GeoTIFF layer boundaries.

    k : float or int, optional
        Perturbation parameter.

        method="shift":
            shift all subsurface layers downward by k.

        method="realization":
            use realization number k.

    method : str, optional
        Perturbation method.

        "shift"        : subtract k from all subsurface layers
        "realization"  : replace layers with realization k

    output_folder : str, optional
        Name of output subfolder.

    Notes
    -----
    top_surface.tif is always copied unchanged.
    """

    folder = Path(folder)

    out_dir = folder / output_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(folder.glob("*.tif"))

    print(f"Found {len(tif_files)} GeoTIFF files")
    print(f"Method: {method}")

    # ------------------------------------------------------------
    # Realization setup
    # ------------------------------------------------------------

    if method == "realization":

        realization_folder = folder.parent / f"realization{k}"

        if not realization_folder.exists():
            raise FileNotFoundError(
                f"Realization folder not found:\n"
                f"{realization_folder}"
            )

        print(f"Using realization: {realization_folder}")

    elif method != "shift":

        raise ValueError(
            f"Unknown method '{method}'"
        )

    # ------------------------------------------------------------
    # Loop files
    # ------------------------------------------------------------

    for fname in tif_files:

        name = fname.stem

        print(f"Processing: {fname.name}")

        with rasterio.open(fname) as src:

            z = src.read(1)
            profile = src.profile.copy()
            nodata = src.nodata

        # --------------------------------------------------------
        # Terrain unchanged
        # --------------------------------------------------------

        if name.lower() == "top_surface":

            z_new = z.copy()

        # --------------------------------------------------------
        # Method: shift
        # --------------------------------------------------------

        elif method == "shift":

            z_new = z.astype(float).copy()

            if nodata is not None:
                mask = z_new == nodata
                z_new[~mask] -= k
            else:
                z_new -= k

        # --------------------------------------------------------
        # Method: realization
        # --------------------------------------------------------

        elif method == "realization":

            replacement_file = realization_folder / fname.name

            if not replacement_file.exists():

                raise FileNotFoundError(
                    f"Missing realization file:\n"
                    f"{replacement_file}"
                )

            with rasterio.open(replacement_file) as src_rep:

                z_new = src_rep.read(1)

        # --------------------------------------------------------
        # Output
        # --------------------------------------------------------

        out_name = f"{name}_perturbed.tif"
        out_path = out_dir / out_name

        with rasterio.open(out_path, "w", **profile) as dst:

            dst.write(z_new, 1)

        print(f"  -> wrote {out_name}")

    print("Done.")