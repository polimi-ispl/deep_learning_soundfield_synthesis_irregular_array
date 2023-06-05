import numpy as np
import tqdm
import scipy
from data_lib import params_linear, params_circular
#import jax.numpy as jnp

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def model_based_synthesis_linear_array(N_lspks, G):
    """
    Computes driving signals using linear loudspeaker array according to
        Bianchi, L., Antonacci, F., Sarti, A., & Tubaro, S. (2016).
        Model-based acoustic rendering based on plane wave decomposition.
        Applied Acoustics, 104, 127-134.

    Args:
        N_lspks: Int, number of loudspeakers
        G: Green function from secondary sources to listening area grid [N_pts, N_ldspks, N_freq]
    Returns:
        N, h, theta_n, theta_min, theta_max, trunc_mod_exp_idx
    """
    # Plane wave directions --> frequency dependent
    radius_lr = 0.5  # Radius of  listening area

    # Truncation order
    M = np.ceil((params_linear.wc / params_linear.c) * np.exp(1) * (radius_lr / 2)).astype(int)
    N = (M + 1) * 2
    trunc_mod_exp_idx = [np.arange(-M[n_f], M[n_f] + 1, 1) for n_f in range(params_linear.N_freqs)]
    # theta_n = np.linspace(0, 2 * np.pi, N)
    theta_min = np.arctan2(params_linear.array_pos[:, 1].min(), params_linear.array_pos[:, 0].min())
    theta_max = np.arctan2(params_linear.array_pos[:, 1].max(), params_linear.array_pos[:, 0].max())
    theta_n = [np.linspace(theta_min, theta_max, n) for n in N]

    # Rendering filters (fixed for setup of choice)
    lambda_ = 1e-2
    G_cp = G[params_linear.idx_lr[params_linear.idx_cp]]  # Green function at control points
    points_cp = params_linear.point[params_linear.idx_lr[params_linear.idx_cp]]

    h = []
    for n_f in tqdm.tqdm(range(len(params_linear.wc))):
        k_n = np.array([np.cos(theta_n[n_f]), np.sin(theta_n[n_f])]).transpose()
        h_temp = np.zeros((N_lspks, N[n_f]), dtype=complex)
        C_m_pwd = np.matmul(
            np.linalg.pinv(np.matmul(G_cp[:, :, n_f].transpose(), G_cp[:, :, n_f]) + lambda_ * np.eye(N_lspks)),
            G_cp[:, :, n_f].transpose())
        for n in range(N[n_f]):
            d = np.exp(1j * (params_linear.wc[n_f] / params_linear.c) * np.matmul(points_cp[:, :2], k_n[n]))
            h_temp[:, n] = np.matmul(C_m_pwd, d)
        h.append(h_temp)
    return N, h, theta_n, theta_min, theta_max, trunc_mod_exp_idx

def model_based_synthesis_circular_array(N_lspks, theta_l,radius_array,radius_lr=1):
    """
    Computes driving signals using circular loudspeaker array according to
        Bianchi, L., Antonacci, F., Sarti, A., & Tubaro, S. (2016).
        Model-based acoustic rendering based on plane wave decomposition.
        Applied Acoustics, 104, 127-134.
    Args:
        N_lspks: Int, number of loudspeakers
        theta_l: secondary sources coordinates in polar form
    Returns:
        N, h, theta_n, theta_min, theta_max, trunc_mod_exp_idx
    """
    # Plane wave directions --> frequency dependent


    # Truncation order
    M = np.ceil((params_circular.wc / params_circular.c) * np.exp(1) * (radius_lr / 2)).astype(int)
    trunc_mod_exp_idx = [np.arange(-M[n_f], M[n_f] + 1, 1) for n_f in range(params_circular.N_freqs)]
    N = (M + 1) * 2
    theta_n = [np.linspace(0, 2 * np.pi, n) for n in N]

    # Rendering filters (fixed for setup of choice)
    h = []
    for n_f in range(len(params_circular.wc)):
        h_temp = np.zeros((N_lspks, N[n_f]), dtype=complex)
        for n_l in range(N_lspks):
            for n in range(N[n_f]):
                num = np.exp(1j * trunc_mod_exp_idx[n_f] * (theta_l[n_l] - theta_n[n_f][n] + (np.pi / 2)))
                den = scipy.special.hankel2(trunc_mod_exp_idx[n_f],
                                            (params_circular.wc[n_f]* radius_array) / (params_circular.c) )
                h_temp[n_l, n] = 4 / (1j * N_lspks) * np.sum(num / den)
        h.append(h_temp)

    return N, h, theta_n, trunc_mod_exp_idx



def herglotz_density_point_source(xs, theta_n, trunc_mod_exp_idx, N, A=1):
    r_z, theta_z = cart2pol(xs[0], xs[1])
    Phi = []
    for n_f in range(params_linear.N_freqs):
        C_m = np.power(1j, -trunc_mod_exp_idx[n_f])  * (1j / 4) * scipy.special.hankel2(
            trunc_mod_exp_idx[n_f], (params_linear.wc[n_f] / params_linear.c) * r_z) * np.exp(
            -1j * theta_z * trunc_mod_exp_idx[n_f])
        Phi_temp = np.zeros((N[n_f]), dtype=complex)
        for n in range(N[n_f]):
            Phi_temp[n] = A * np.sum(C_m * np.exp(1j * trunc_mod_exp_idx[n_f] * theta_n[n_f][n]))
        Phi.append(Phi_temp)
    return Phi


