# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:30:02 2026

@author: Q362849
"""

import os
import glob
import rasterio

def perturb_layer_boundaries(folder, k, output_folder="perturbed"):
    """
    Perturb all layer boundary GeoTIFFs except the top surface.

    Parameters
    ----------
    folder : str
        Folder containing GeoTIFF layer boundaries.
    k : float
        Scalar shift applied to all subsurface layers.
        New layer = old layer - k
    output_folder : str, optional
        Name of output subfolder written inside `folder`.

    Notes
    -----
    - top_surface.tif is copied unchanged
    - all other .tif files are shifted downward by k
    - outputs are written as:
          <original_name>_perturbed.tif
    """

    # ------------------------------------------------------------
    # Create output folder
    # ------------------------------------------------------------
    out_dir = os.path.join(folder, output_folder)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Find all tif files
    # ------------------------------------------------------------
    tif_files = sorted(glob.glob(os.path.join(folder, "*.tif")))

    print(f"Found {len(tif_files)} GeoTIFF files")

    for fname in tif_files:

        base = os.path.basename(fname)
        name = os.path.splitext(base)[0]

        print(f"Processing: {base}")

        # --------------------------------------------------------
        # Read raster
        # --------------------------------------------------------
        with rasterio.open(fname) as src:

            z = src.read(1)
            profile = src.profile.copy()
            nodata = src.nodata

            # ----------------------------------------------------
            # Keep terrain unchanged
            # ----------------------------------------------------
            if name.lower() == "top_surface":

                z_new = z.copy()

            else:

                z_new = z.astype(float).copy()

                # Avoid perturbing nodata cells
                if nodata is not None:
                    mask = z_new == nodata
                    z_new[~mask] -= k
                else:
                    z_new -= k

            # ----------------------------------------------------
            # Output filename
            # ----------------------------------------------------
            out_name = f"{name}_perturbed.tif"
            out_path = os.path.join(out_dir, out_name)

            # ----------------------------------------------------
            # Write GeoTIFF
            # ----------------------------------------------------
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(z_new, 1)

        print(f"  -> wrote {out_name}")

    print("Done.")