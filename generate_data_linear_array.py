import numpy as np
import matplotlib.pyplot as plt
import sfs
import tqdm
import scipy
from data_lib import soundfield_generation as sg
from data_lib import params_linear
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate data for linear array setup')
    parser.add_argument('--gt_soundfield', type=bool, help='compute ground truth soundfield',
                        default=True)
    parser.add_argument('--n_missing', type=int, help='number missing loudspeakers',
                        default=0)
    parser.add_argument('--dataset_path', type=str, help='number missing loudspeakers',
                        default='/nas/home/lcomanducci/soundfield_synthesis/dataset/linear_array')
    args = parser.parse_args()
    propagate_filters = True
    eval_points = False
    dataset_path = args.dataset_path

    # Setup
    point = params_linear.point
    N_pts = len(point)

    # Secondary Sources Green function
    green_function_sec_sources_path = 'green_function_sec_sources_nl_' + str(params_linear.N_lspks) + '.npy'
    if os.path.exists(os.path.join(dataset_path, green_function_sec_sources_path)):
        G = np.load(os.path.join(dataset_path, green_function_sec_sources_path))
    else:
        G = np.zeros((N_pts, params_linear.N_lspks, params_linear.N_freqs),
                     dtype=complex)
        for n_p in tqdm.tqdm(range(N_pts)):
            hankel_factor_1 = np.tile(params_linear.wc / params_linear.c, (params_linear.N_lspks, 1))
            hankel_factor_2 = np.tile(np.linalg.norm(point[n_p] - params_linear.array_pos, axis=1),
                                      reps=(params_linear.N_freqs, 1)).transpose()
            G[n_p, :, :] = (1j / 4) * scipy.special.hankel2(0, hankel_factor_1 * hankel_factor_2)
        np.save(os.path.join(dataset_path, green_function_sec_sources_path), G)

    # Check if array and grid points are equal
    for n_p in range(N_pts):
        if np.sum(np.linalg.norm(point[n_p] - params_linear.array_pos, axis=1) == 0) > 0:
            print(str(n_p))

    # Model-based acoustic rendering based on plane wave decomposition
    N_missing = args.n_missing
    N_lspks = params_linear.N_lspks - N_missing
    if N_missing > 0:
        lspk_config_path = 'lspk_config_nl_' + str(params_linear.N_lspks) + '_missing_' + str(N_missing) + '.npy'
        lspk_config_path_global = os.path.join(dataset_path, 'setup', lspk_config_path)
        if os.path.exists(lspk_config_path_global):
            idx_missing = np.load(lspk_config_path_global)
            print('Loaded existing mic config')
        else:
            idx_missing = np.random.choice(np.arange(params_linear.N_lspks), size=N_missing, replace=False)
            np.save(lspk_config_path_global, idx_missing)
        theta_l = np.delete(params_linear.theta_l, idx_missing)
        G = np.delete(G, idx_missing, axis=1)

    N, h, theta_n, theta_min, theta_max, trunc_mod_exp_idx = sg.model_based_synthesis_linear_array(N_lspks, G)

    if eval_points:
        N_pts = len(params_linear.idx_cp)
        G = G[params_linear.idx_lr[params_linear.idx_cp]]
        point = params_linear.point_cp

    P_gt = np.zeros((params_linear.N_sources, N_pts, params_linear.N_freqs), dtype=complex)
    d_array = np.zeros((params_linear.N_sources, N_lspks, params_linear.N_freqs), dtype=complex)
    print('Now we cycle through sources')
    for n_s in tqdm.tqdm(range(params_linear.N_sources)):
        xs = params_linear.src_pos_train[n_s, :]
        # Herglotz density function - point source
        Phi = sg.herglotz_density_point_source(xs, theta_n, trunc_mod_exp_idx, N, A=1)

        # Multiply filters for herglotz density function
        if propagate_filters:
            P = np.zeros((N_pts, len(params_linear.wc)), dtype=complex)
        for n_f in range(params_linear.N_freqs):
            d_array[n_s, :, n_f] = ((theta_max - theta_min) / (N[n_f] * 2 * np.pi)) * np.matmul(np.expand_dims(Phi[n_f], axis=0), h[n_f].transpose())#np.sum(Phi[n_f] * h[n_f], axis=1)

        if propagate_filters:
            for n_p in range(N_pts):
                P[n_p, :] = np.sum(G[n_p] * d_array[n_s], axis=0)

        if args.gt_soundfield:
            for n_f in range(params_linear.N_freqs):
                P_gt[n_s, :, n_f] = (1j / 4) * \
                                    scipy.special.hankel2(0,
                                                          (params_linear.wc[n_f] / params_linear.c) *
                                                          np.linalg.norm(point[:, :2] - xs, axis=1))

    if args.gt_soundfield:
        np.save(os.path.join(dataset_path, 'gt_soundfield_train.npy'), P_gt)
    np.save(os.path.join(dataset_path,
                         'filters_config_nl_' + str(params_linear.N_lspks) + '_missing_' + str(N_missing) + '.npy'), d_array)


if __name__ == '__main__':
    main()


