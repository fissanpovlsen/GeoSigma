import numpy as np
from geosigma import local_kriging_setup_img

# Fake test data
nx, ny = 10, 8
GLOBAL = {
    "xx_norm": np.linspace(0,1,nx*ny),
    "yy_norm": np.linspace(0,1,nx*ny),
    "img_dobs": np.random.rand(nx*ny),
    "img_unc": np.ones(nx*ny) * 0.1
}
i_buf = np.ones(nx*ny, dtype=bool)
ip_buf = np.random.rand(nx*ny) > 0.5

G, Cm, Cd, d_obs, m0 = local_kriging_setup_img(
    GLOBAL, i_buf, ip_buf, curvariance=1.0, currange=5.0, cor_noise_rat=0.7, plot=True
)
