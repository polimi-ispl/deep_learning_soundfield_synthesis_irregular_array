import matplotlib.pyplot as plt
import numpy as np
import sfs
from data_lib import params_linear, params_circular
from skimage.metrics import structural_similarity as ssim

# Do plots using latex
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    'font.size': 20})

def plot_soundfield(cmap, P, n_f, selection, axis_label_size, tick_font_size, save_path, array_type, plot_ldspks=True, do_norm=True):
    """
    Args:
        cmap: string, colormap
        P: complex soundfield [N_pts, N_freqs] (only real part is shown)
        n_f: int, frequency idx
        selection: bool, active loudspeaker idx
        axis_label_size: Int, size of labels on the axes
        tick_font_size: Int, size of ticks on the axes
        save_path: string, path where plot is saved
        array_type: string, 'linear' for linear array setup, 'circular' for circular array setup
        plot_ldspks: bool, if True plot loudspeakers
        do_norm: bool, normalize the plot between -1 and 1
    Returns:
        Nothing, shows the plot and saves it in save_path

    """

    if array_type == 'linear':
        params = params_linear
    elif array_type == 'circular':
        params = params_circular
    else:
        print('Error')

    figure = plt.figure(figsize=(20, 20))
    if do_norm:
        im = sfs.plot2d.amplitude(np.reshape(P[:, n_f], (params.N_sample, params.N_sample)),
                                  params.grid, xnorm=[0, 0, 0], cmap=cmap, vmin=-1.0, vmax=1.0, colorbar=False)
    else:
        im = sfs.plot2d.amplitude(np.reshape(P[:, n_f], (params.N_sample, params.N_sample)),
                                  params.grid,  cmap=cmap, colorbar=False, vmin=P[:, n_f].min(), vmax=P[:, n_f].max(), xnorm=None)
    if plot_ldspks:
        sfs.plot2d.loudspeakers(params.array.x[selection], -params.array.n[selection], a0=1, size=0.18)
    plt.xlabel('$x [m]$', fontsize=axis_label_size), plt.ylabel('$y [m]$', fontsize=axis_label_size)
    plt.tick_params(axis='both', which='major', labelsize=tick_font_size)
    cbar = plt.colorbar(im, fraction=0.046)
    cbar.ax.tick_params(labelsize=tick_font_size)
    # cbar.set_label('$NRE~[\mathrm{dB}]$',fontsize=tick_font_size))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def normalize(x):
    """
    Normalizes a tensor between 0 and 1
    Args:
        x, Tensor
    Returns:
        x_norm, normalized version of x
    """
    min_x = x.min()
    max_x = x.max()
    x_norm = (x - min_x)/(max_x-min_x)
    return x_norm

def nmse(P_hat, P_gt, type='freq'):
    """
    Computes NMSE between ground truth and estimated soundfield
    Args:
        P_hat: np.array, complex estimated soundfield [N_points, N_freqs]
        P_gt:  np.array, complex ground truth soundfield [N_points, N_freqs]
        type: 'freq' --> averages over all listening points (nmse per each frequencies), otherwise does not average
    Returns
        NMSE: Normalized Mean Squared Error (or Normalized Reproduction Error)
    """
    if type =='freq':
        return np.mean((np.power(np.abs(P_hat[:, :] - P_gt[:, :]), 2) / np.power(np.abs(P_gt[:, :]), 2)), axis=0)
    else:
        return np.power(np.abs(P_hat[:, :] - P_gt[:, :]), 2) / np.power(np.abs(P_gt[:, :]), 2)

def ssim_abs(P_hat, P_gt):
    """
    Computes structural similarity index (SSIM) between normalized
    between ground truth and estimated soundfield (--> range=1) at one frequency

        Z. Wang, A. Bovik, H. Sheikh, and E. Simoncelli, “Image quality assessment: From error visibility to
        structural similarity,” IEEE Trans.Image Process., vol. 13, no. 4, pp. 600–612, Apr. 2004

    Args:
        P_hat: np.array, complex estimated soundfield at one frequency [N_points]
        P_gt:  np.array, complex ground truth soundfield at one frequency [N_points]
    Returns
        SSIM: Normalized Mean Squared Error (or Normalized Reproduction Error)
    """
    P_hat = normalize(np.abs(P_hat))
    P_gt = normalize(np.abs(P_gt))
    return ssim(P_gt, P_hat, data_range=1)

def ssim_freq(P_hat, P_gt):
    """
       Computes structural similarity index (SSIM) between normalized
       between ground truth and estimated soundfield (--> range=1) over N_freqs frequencies

           Z. Wang, A. Bovik, H. Sheikh, and E. Simoncelli, “Image quality assessment: From error visibility to
           structural similarity,” IEEE Trans.Image Process., vol. 13, no. 4, pp. 600–612, Apr. 2004

       Args:
        P_hat: np.array, complex estimated soundfield [N_points, N_freqs]
        P_gt:  np.array, complex ground truth soundfield [N_points, N_freqs]
       Returns
           SSIM: Normalized Mean Squared Error (or Normalized Reproduction Error)
       """
    ssim_freq_array = np.zeros(params_linear.N_freqs)
    for n_f in range(params_linear.N_freqs):
        ssim_freq_array[n_f] = ssim_abs(P_hat[:, n_f], P_gt[:, n_f])
    return ssim_freq_array