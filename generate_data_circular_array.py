import numpy as np
import os
import scipy
import tqdm
import argparse
from data_lib import params_circular
from data_lib import soundfield_generation as sg, results_utils

def main():
    # Arguments parse
    parser = argparse.ArgumentParser(description='Generate data for circular array setup')
    parser.add_argument('--gt_soundfield', type=bool, help='compute ground truth soundfield',
                        default=False)
    parser.add_argument('--n_missing', type=int, help='number missing loudspeakers',
                        default=32)
    parser.add_argument('--dataset_path', type=str, help='number missing loudspeakers',
                        default='/nas/home/lcomanducci/soundfield_synthesis/dataset/circular_array')
    args = parser.parse_args()

    eval_points = True
    propagate_filters = False
    dataset_path = args.dataset_path

    # Setup
    # Grid of points where we actually compute the soundfield
    point = params_circular.point
    N_pts = len(point)

    # Secondary Sources Green function
    green_function_sec_sources_path = 'green_function_sec_sources_nl_' + str(params_circular.N_lspks) + '_r_' + str(params_circular.radius) + '.npy'
    if os.path.exists(os.path.join(dataset_path, green_function_sec_sources_path)):
        G = np.load(os.path.join(dataset_path, green_function_sec_sources_path))
    else:
        G = np.zeros((N_pts, params_circular.N_lspks, params_circular.N_freqs),
                     dtype=complex)
        for n_p in tqdm.tqdm(range(N_pts)):
            hankel_factor_1 = np.tile(params_circular.wc / params_circular.c, (params_circular.N_lspks, 1))
            hankel_factor_2 = np.tile(np.linalg.norm(point[n_p] - params_circular.array_pos, axis=1), reps=(params_circular.N_freqs, 1)).transpose()
            G[n_p, :, :] = (1j / 4) * scipy.special.hankel2(0, hankel_factor_1*hankel_factor_2)
        np.save(os.path.join(dataset_path, green_function_sec_sources_path), G)

    for n_p in range(N_pts):
        if np.sum(np.linalg.norm(point[n_p] - params_circular.array_pos, axis=1) == 0) > 0:
            print(str(n_p))

    # Model-based acoustic rendering based on plane wave decomposition
    # Missing loudspeakers
    N_missing = args.n_missing
    lspk_config_path = 'lspk_config_nl_' + str(params_circular.N_lspks) + '_missing_' + str(N_missing) + '.npy'
    lspk_config_path_global = os.path.join(dataset_path, 'setup', lspk_config_path)
    if os.path.exists(lspk_config_path_global):
        idx_missing = np.load(lspk_config_path_global)
        print('Loaded existing mic config')
    else:
        idx_missing = np.random.choice(np.arange(params_circular.N_lspks), size=N_missing, replace=False)
        np.save(lspk_config_path_global, idx_missing)
        print('CONFIG NEW')
    N_lspks = params_circular.N_lspks - N_missing
    theta_l = np.delete(params_circular.theta_l, idx_missing)
    G = np.delete(G, idx_missing, axis=1)

    # Driving signals
    N, h, theta_n, trunc_mod_exp_idx = sg.model_based_synthesis_circular_array(N_lspks, theta_l)

    # Array holding training data
    if eval_points:
        N_pts = len(params_circular.idx_lr)
        G = G[params_circular.idx_lr]
        point = params_circular.point_lr

    P_gt = np.zeros((len(params_circular.src_pos_train), N_pts, params_circular.N_freqs), dtype=complex)
    d_array = np.zeros((len(params_circular.src_pos_train), N_lspks, params_circular.N_freqs), dtype=complex)

    for n_s in tqdm.tqdm(range(len(params_circular.src_pos_train))):
        xs = params_circular.src_pos_train[n_s]
        # Herglotz density function - point source
        Phi = sg.herglotz_density_point_source(xs, theta_n, trunc_mod_exp_idx, N, A=1)

        # Multiply filters for herglotz density function
        if propagate_filters:
            P = np.zeros((N_pts, len(params_circular.wc)), dtype=complex)
        for n_f in range(params_circular.N_freqs):
            d_array[n_s, :, n_f] = (1 / N[n_f]) * np.matmul(np.expand_dims(Phi[n_f], axis=0), h[n_f].transpose())

        if propagate_filters:
            for n_p in range(N_pts):
                P[n_p, :] = np.sum(G[n_p]*d_array[n_s], axis=0)

        # Ground truth source
        if args.gt_soundfield:
            for n_f in range(params_circular.N_freqs):
                P_gt[n_s, :, n_f] = (1j / 4) * \
                               scipy.special.hankel2(0,
                                                     (params_circular.wc[n_f] / params_circular.c) *
                                                     np.linalg.norm(point[:, :2] - xs, axis=1))
    if args.gt_soundfield:
        np.save(os.path.join(dataset_path, 'gt_soundfield_train.npy'), P_gt)
    np.save(os.path.join(dataset_path,
                         'filters_config_nl_' + str(params_circular.N_lspks) + '_missing_' + str(N_missing) + '.npy'), d_array)


if __name__ == '__main__':
    main()
