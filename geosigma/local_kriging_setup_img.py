# geosigma/local_kriging_setup_img.py
import numpy as np
from .precal_cov import precal_cov
import matplotlib.pyplot as plt

def local_kriging_setup_img(GLOBAL, i_buf, ip_buf, curvariance, currange, cor_noise_rat, plot=False):
    """
    Sets up input matrices for local kriging:
    G, Cm, Cd, d_obs, m0
    
    Parameters
    ----------
    GLOBAL : dict
        Contains keys 'xx_norm', 'yy_norm', 'img_dobs', 'img_unc'
        All values should be NumPy arrays.
    i_buf : array-like, bool or 0/1
        Buffer mask for total nodes
    ip_buf : array-like, bool or 0/1
        Buffer mask for observed nodes
    curvariance : float
        Sill of the covariance
    currange : float
        Range parameter for isotropic Gaussian covariance
    cor_noise_rat : float
        Correlation ratio for Cd
    plot : bool, optional
        If True, plot G, Cm, Cd matrices for sanity check
    
    Returns
    -------
    G : ndarray
    Cm : ndarray
    Cd : ndarray
    d_obs : ndarray
    m0 : float
    """

    # --- Local buffers ---
    i_buf_local = np.array(i_buf, dtype=bool)
    ip_buf_local = np.array(ip_buf, dtype=bool)
    
    # --- G matrix ---
    G = np.eye(np.sum(i_buf_local), dtype=float)
    G = G[ip_buf_local, :]

    # --- Model covariance Cm ---
    coords_cm = np.column_stack((
        GLOBAL["xx_norm"][i_buf_local] * 100,
        GLOBAL["yy_norm"][i_buf_local] * 100
    ))
    statmod_var = f"{curvariance} Gau({currange})"
    Cm, _ = precal_cov(coords_cm, coords_cm, statmod_var)
    Cm = np.asarray(Cm, dtype=float)

    # --- Data covariance Cd ---
    coords_cd = np.column_stack((
        GLOBAL["xx_norm"][ip_buf_local] * 100,
        GLOBAL["yy_norm"][ip_buf_local] * 100
    ))
    statmod_noi = f"{curvariance} Gau({currange})"
    Cd_shape, _ = precal_cov(coords_cd, coords_cd, statmod_noi)
    Cd_shape = np.asarray(Cd_shape, dtype=float)

    img_unc_vec = np.asarray(GLOBAL["img_unc"][ip_buf_local], dtype=float)
    Cd_diag = np.diag(img_unc_vec)

    # Build full Cd
    Cd = Cd_diag @ (cor_noise_rat * Cd_shape + (1 - cor_noise_rat) * np.eye(len(Cd_shape))) @ Cd_diag

    # --- Observed data ---
    d_obs = np.asarray(GLOBAL["img_dobs"][ip_buf_local], dtype=float)

    # --- Prior mean ---
    m0 = 0.0

    # --- Optional plotting for sanity check ---
    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(15,5))
        im0 = axs[0].imshow(G, origin='lower', cmap='viridis')
        axs[0].set_title("G matrix")
        plt.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(Cm, origin='lower', cmap='viridis')
        axs[1].set_title("Cm (model covariance)")
        plt.colorbar(im1, ax=axs[1])

        im2 = axs[2].imshow(Cd, origin='lower', cmap='viridis')
        axs[2].set_title("Cd (data covariance)")
        plt.colorbar(im2, ax=axs[2])

        plt.show()

    return G, Cm, Cd, d_obs, m0
