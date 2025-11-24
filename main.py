""" Main function for geostatistical modelling code """

### Import relevant python packages
import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

### Import functions from python scripts
from import_complexitymap import import_complexitymap
from get_PACES_theme import get_PACES_theme
from get_GAMMALOG_theme import get_GAMMALOG_theme
from get_RESLOG_theme import get_RESLOG_theme
from get_REFSEIS_theme import get_REFSEIS_theme
from get_fewTEM_theme import get_fewTEM_theme
from get_manyTEM_theme import get_manyTEM_theme
from get_tTEM_theme import get_tTEM_theme
from get_MEP_theme import get_MEP_theme
from get_SkyTEM_theme import get_SkyTEM_theme
from get_PACEP_theme import get_PACEP_theme

### Additional functions
def make_test_area():
    nx, ny = 100, 100
    Nlay = 12
    NPL = 4
    Npreq = 3
    region = 'Jylland'

    #PACES TEST AREA
    #XS = np.arange(546050, 579051, nx)
    #YS = np.arange(6200050, 6237051, ny)

    #GAMMA LOG TEST AREA
    XS = np.arange(527050, 545051, nx)
    YS = np.arange(6141050, 6171051, ny)
    terrain = np.random.rand(ny, nx)


    return XS, YS, terrain, region, Npreq, NPL, Nlay

import struct


def debug_grd_file(filename):
    with open(filename, 'rb') as f:
        print("Reading GRD File")

        # Read file type
        file_type = f.read(4).decode('ascii')
        print(f"FileType - {file_type}")
        if file_type != "DSRB":
            raise ValueError("File format not supported!")

        # Read header size and version
        header_size = struct.unpack('i', f.read(4))[0]
        header_version = struct.unpack('i', f.read(4))[0]
        print(f"Header Size: {header_size}, Header Version: {header_version}")

        # Read section name
        section_name = f.read(4).decode('ascii')
        print(f"Section Name: {section_name}")

        # Read section length
        sec2 = struct.unpack('i', f.read(4))[0]
        print(f"Section Length: {sec2}")

        # Grid size
        grid_rows, grid_cols = struct.unpack('ii', f.read(8))
        print(f"Grid Size: {grid_rows} rows, {grid_cols} cols")

        # Grid corner location
        xLL, yLL = struct.unpack('dd', f.read(16))
        print(f"xLL: {xLL}, yLL: {yLL}")

        # Grid resolution
        x_res, y_res = struct.unpack('dd', f.read(16))
        print(f"x-size: {x_res}m, y-size: {y_res}m")

        # Min and Max Z
        min_z, max_z = struct.unpack('dd', f.read(16))
        print(f"Min. Z: {min_z}m, Max. Z: {max_z}m")

        # Grid rotation
        grid_rotation = struct.unpack('dd', f.read(16))

        # Blank value
        blank_value = struct.unpack('d', f.read(8))[0]

        # Read next 32 bytes to inspect alignment
        next_bytes = f.read(32)
        print("Next 32 bytes after blank value:", next_bytes)

        # Try reading data section name and length
        f.seek(f.tell() - 32)  # rewind to start of those 32 bytes
        section_name2 = f.read(4).decode('ascii')
        print(f"Data Section Name: {section_name2}")

        data_section_length_bytes = f.read(4)
        data_section_length = struct.unpack('i', data_section_length_bytes)[0]
        print(f"Data Section Length: {data_section_length} bytes ({data_section_length / 1024:.2f} KB)")


def plot_paces_theme(Grid, layers=None, mask_value=100000):
    """
    Plot selected layers from the PACES theme grid, masking out specific values.
    
    Parameters:
        Grid (numpy.ndarray): 3D array (ny, nx, nlayers)
        layers (list): List of layer indices to plot. Defaults to first, middle, last.
        mask_value (float): Value to mask out (default: 100000)
    """
    # Mask out unwanted values
    masked_Grid = np.ma.masked_where(Grid == mask_value, Grid)

    ny, nx, nlayers = Grid.shape
    if layers is None:
        layers = [0, nlayers // 2, nlayers - 1]

    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 6))
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        im = ax.imshow(masked_Grid[:, :, layer], cmap=cmc.batlow, origin='lower',vmax=6)
        ax.set_title(f'Layer {layer}')
        ax.set_xlabel('X index')
        ax.set_ylabel('Y index')
        fig.colorbar(im, ax=ax, orientation='vertical')

    plt.tight_layout()
    plt.show()


def plot_gammalog_theme(Grid, layers=None, mask_value=100000):
    """
    Plot selected layers from the GAMMALOG theme grid, masking out specific values.

    Parameters:
        Grid (numpy.ndarray): 3D array (ny, nx, nlayers)
        layers (list): List of layer indices to plot. Defaults to first, middle, last.
        mask_value (float): Value to mask out (default: 100000)
    """
    # Mask out unwanted values
    masked_Grid = np.ma.masked_where(Grid == mask_value, Grid)

    ny, nx, nlayers = Grid.shape
    if layers is None:
        layers = [0, nlayers // 2, nlayers - 1]

    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 6))
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        im = ax.imshow(masked_Grid[:, :, layer], cmap=cmc.batlow, origin='lower', vmax=6)
        ax.set_title(f'GAMMALOG Theme - Layer {layer}')
        ax.set_xlabel('X index')
        ax.set_ylabel('Y index')
        fig.colorbar(im, ax=ax, orientation='vertical', label='Uncertainty')

    plt.tight_layout()
    plt.show()

def plot_theme(Grid, theme_name="Theme", layers=None, mask_value=100000, cmap=cmc.batlow, vmax=None):
    """
    Plot selected layers from any theme grid, masking out specific values.

    Parameters:
        Grid (numpy.ndarray): 3D array (ny, nx, nlayers)
        theme_name (str): Name of the theme for titles
        layers (list): List of layer indices to plot. Defaults to first, middle, last.
        mask_value (float): Value to mask out (default: 100000)
        cmap: Colormap (default: cmcrameri.batlow)
        vmax: Optional max value for color scaling
    """
    # Mask out unwanted values
    masked_Grid = np.ma.masked_where(Grid == mask_value, Grid)

    ny, nx, nlayers = Grid.shape
    if layers is None:
        layers = [0, nlayers // 2, nlayers - 1]

    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 6))
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        im = ax.imshow(masked_Grid[:, :, layer], cmap=cmap, origin='lower', vmax=vmax)
        ax.set_title(f'{theme_name} - Layer {layer}')
        ax.set_xlabel('X index')
        ax.set_ylabel('Y index')
        fig.colorbar(im, ax=ax, orientation='vertical', label='Uncertainty')

    plt.tight_layout()
    plt.show()

### Start main
if __name__ == "__main__":

    print('Make test area')
    XS, YS, terrain, region, Npreq, NPL, Nlay = make_test_area()

    print('Import complexity map')
    complexity = import_complexitymap(XS,YS,region)

    print('Preparing Geophysics Region-Specific Uncertainty Themes')
    print('get PACES theme')
    #PACES_themes, PACES_c = get_PACES_theme(XS, YS, terrain, complexity, region, 1, Npreq, NPL, Nlay)
    #plot_theme(PACES_themes, theme_name="REFSEIS", layers=[6,7,8,9,10], mask_value=100000, cmap=cmc.batlow, vmax=6)
    print('get_PACES_theme...Done')

    print('get_GAMMALOG_theme...')
    #GAMMALOG_themes,GAMMALOG_c = get_GAMMALOG_theme(XS, YS, terrain, complexity, region, 1, Npreq, NPL, Nlay)
    #plot_gammalog_theme(GAMMALOG_themes, layers=[5,6,7,8])
    print('get_GAMMALOG_theme...Done')

    print('get_RESLOG_theme...')
    #RESLOG_themes,RESLOG_c = get_RESLOG_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay)
    #plot_theme(RESLOG_themes, theme_name="RESLOG", layers=[5,6,7,8], mask_value=100000, cmap=cmc.batlow, vmax=6)
    print('get_RESLOG_theme...Done')

    print('get_REFSEIS_theme...')
    #REFSEIS_themes,REFSEIS_c = get_REFSEIS_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay)
    #plot_theme(REFSEIS_themes, theme_name="REFSEIS", layers=[6,7,8,9], mask_value=100000, cmap=cmc.batlow, vmax=100)
    print('get_REFSEIS_theme...Done')

    print('get_fewTEM_theme...')
    #fewTEM_themes, fewTEM_c = get_fewTEM_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay)
    #plot_theme(fewTEM_themes, theme_name="fewTEM", layers=[6,7,8,9], mask_value=100000, cmap=cmc.batlow, vmax=100)
    print('get_fewTEM_theme...Done')

    print('get_manyTEM_theme...')
    #manyTEM_themes, manyTEM_c = get_manyTEM_theme(XS,YS,terrain,complexity,region,1,Npreq,NPL,Nlay)
    #plot_theme(manyTEM_themes, theme_name="manyTEM", layers=[4,10,15,20,30], mask_value=100000, cmap=cmc.batlow, vmax=1000)
    print('get_manyTEM_theme...Done')

    print('get_tTEM_theme...')
    #tTEM_themes, tTEM_c = get_tTEM_theme(XS, YS, terrain, complexity, region, 1, Npreq, NPL, Nlay)
    #plot_theme(tTEM_themes, theme_name="tTEM", layers=[4,5,6,7,8,9], mask_value=100000, cmap=cmc.batlow, vmax=1000)
    print('get_tTEM_theme...Done')


    print('get_MEP_theme...')
    #MEP_themes, MEP_c = get_MEP_theme(XS, YS, terrain, complexity, region, 1, Npreq, NPL, Nlay)
    #plot_theme(MEP_themes, theme_name="MEP", layers=[9,10,11], mask_value=100000, cmap=cmc.batlow, vmax=5)
    print('get_MEP_theme...Done')

    print('get_SkyTEM_theme...')
    #SkyTEM_themes, SkyTEM_c = get_SkyTEM_theme(XS, YS, terrain, complexity, region, 1, Npreq, NPL, Nlay)
    #plot_theme(SkyTEM_themes, theme_name="SkyTEM", layers=[5,6,7], mask_value=100000, cmap=cmc.batlow, vmax=20)
    print('get_SkyTEM_theme...Done')


    print('get_PACEP_theme...')
    PACEP_themes, PACEP_c = get_PACEP_theme(XS, YS, terrain, complexity, region, 1, Npreq, NPL, Nlay)
    plot_theme(PACEP_themes, theme_name="PACEP", layers=[6,7,8,9], mask_value=100000, cmap=cmc.batlow, vmax=20)
    print('get_PACEP_theme...Done')