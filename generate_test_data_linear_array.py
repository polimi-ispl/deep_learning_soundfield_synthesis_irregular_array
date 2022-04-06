import numpy as np
import scipy
import tqdm
import argparse
import tensorflow as tf
from data_lib import params_linear
from data_lib import soundfield_generation as sg, results_utils
import jax.numpy as jnp
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def main():
    # Arguments parse
    parser = argparse.ArgumentParser(description='Generate data for linear array setup')
    parser.add_argument('--dataset_path', type=str, help="Base Data Directory", default='/nas/home/lcomanducci/soundfield_synthesis/dataset/test/linear_array')
    parser.add_argument('--models_path',type=str, help='Deep learning models folder', default='/nas/home/lcomanducci/soundfield_synthesis/models/linear_array/')
    parser.add_argument('--gt_soundfield', type=bool, help='compute ground truth soundfield', default=True)
    parser.add_argument('--pm', type=bool, help='compute pressure matching', default=True)
    parser.add_argument('--pwd', type=bool, help='compute model-based acoustic rendering', default=True)
    parser.add_argument('--pwd_cnn', type=bool, help='compute model-based acoustic rendering + CNN', default=True)
    parser.add_argument('--n_missing', type=int, help='number missing loudspeakers',
                        default=48)
    eval_points = False
    PLOT = True
    args = parser.parse_args()
    dataset_path = args.dataset_path

    # Grid of points where we actually compute the soundfield
    point = params_linear.point
    N_pts = len(point)

    # Load green function secondary sources --> eval points (it is in train directory since it is the same)
    dataset_path_train = '/nas/home/lcomanducci/soundfield_synthesis/dataset/linear_array'
    green_function_sec_sources_path = 'green_function_sec_sources_nl_' + str(params_linear.N_lspks) + '.npy'
    G = np.load(os.path.join(dataset_path_train, green_function_sec_sources_path))

    # Load Missing loudspeakers configuration
    lspk_config_path = 'lspk_config_nl_' + str(params_linear.N_lspks) + '_missing_' + str(args.n_missing) + '.npy'
    lspk_config_path_global = os.path.join(dataset_path_train, 'setup', lspk_config_path)
    idx_missing = np.load(lspk_config_path_global)
    N_lspks = params_linear.N_lspks - args.n_missing
    G = np.delete(G, idx_missing, axis=1)

    # Let's precompute what we need in order to apply the selected models
    # Load pwd_cnn deep learning model
    if args.pwd_cnn:
        model_name = 'model_linear_config_nl_'+str(params_linear.N_lspks)+'_missing_'+str(args.n_missing)
        network_model = tf.keras.models.load_model(os.path.join(args.models_path, model_name))

        # Driving signals - Model-based acoustic rendering
        N, h, theta_n, theta_min, theta_max, trunc_mod_exp_idx = sg.model_based_synthesis_linear_array(N_lspks, G)

    if args.pm:
        lambda_ = 1e-2
        G_cp = G[params_linear.idx_lr[params_linear.idx_cp]]  # Green function at control points
        # points_cp = params_linear.point[params_linear.idx_lr[params_linear.idx_cp]]
        C_pm = np.zeros((N_lspks, len(params_linear.idx_cp), params_linear.N_freqs), dtype=complex)
        for n_f in tqdm.tqdm(range(len(params_linear.wc))):
            C_pm[:, :, n_f] = np.matmul(jnp.linalg.pinv(np.matmul(G_cp[:, :, n_f].transpose(), G_cp[:, :, n_f]) + lambda_ * np.eye(N_lspks)),G_cp[:, :, n_f].transpose())

    if eval_points:
        N_pts = len(params_linear.idx_lr)
        G = G[params_linear.idx_lr]
        point = params_linear.point_lr

    nmse_pwd = np.zeros((params_linear.n_src_test,  params_linear.N_freqs))
    nmse_pwd_cnn, nmse_pwd_pm = np.zeros_like(nmse_pwd), np.zeros_like(nmse_pwd)
    ssim_pwd, ssim_pwd_cnn, ssim_pwd_pm = np.zeros_like(nmse_pwd),  np.zeros_like(nmse_pwd),  np.zeros_like(nmse_pwd)

    for n_s in tqdm.tqdm(range(params_linear.n_src_test)):
        if PLOT:
            n_s = 1900

        P_gt = np.zeros((N_pts, params_linear.N_freqs), dtype=complex)
        P_pwd = np.zeros_like(P_gt, dtype=complex)
        P_pwd_cnn = np.zeros_like(P_gt, dtype=complex)
        P_pwd_pm = np.zeros_like(P_gt, dtype=complex)
        xs = params_linear.src_pos_test[:, n_s]

        if args.gt_soundfield:
            for n_f in range(params_linear.N_freqs):
                hankel_arg = (params_linear.wc[n_f] / params_linear.c) * np.linalg.norm(point[:, :2] - xs, axis=1)
                P_gt[ :, n_f] = (1j / 4) * scipy.special.hankel2(0, hankel_arg)

        if args.pwd:
            Phi = sg.herglotz_density_point_source(xs, theta_n, trunc_mod_exp_idx, N, A=1)

            # Multiply filters for herglotz density function
            d_array = np.zeros((N_lspks, params_linear.N_freqs), dtype=complex)
            for n_f in range(params_linear.N_freqs):
                d_array[:, n_f] = ((theta_max - theta_min) / (N[n_f] * 2 * np.pi)) * np.matmul(np.expand_dims(Phi[n_f], axis=0),h[n_f].transpose())
            for n_p in range(N_pts):
                P_pwd[n_p, :] = np.sum(G[n_p] * d_array, axis=0)

        if args.pwd_cnn:
            d_array_cnn = network_model.predict(
                np.expand_dims(np.concatenate([np.real(d_array), np.imag(d_array)], axis=0), axis=[0, -1]).astype('float32'))[0, :, :, 0].astype('float64')

            d_array_cnn_complex = d_array_cnn[:int(d_array_cnn.shape[0] / 2)] + (1j * d_array_cnn[int(d_array_cnn.shape[0] / 2):])
            for n_p in range(N_pts):
                P_pwd_cnn[n_p, :] = np.sum(G[n_p] * d_array_cnn_complex, axis=0)

        if args.pm:
            d_pm = np.zeros((N_lspks, params_linear.N_freqs),dtype=complex)
            for n_f in range(params_linear.N_freqs):
                if eval_points:
                    d_pm[:, n_f] = np.matmul(C_pm[:, :, n_f], P_gt[params_linear.idx_cp,n_f])
                else:
                    d_pm[:, n_f] = np.matmul(C_pm[:, :, n_f], P_gt[params_linear.idx_lr[params_linear.idx_cp],n_f])

            for n_p in range(N_pts):
                P_pwd_pm[ n_p, :] = np.sum(G[n_p] * d_pm, axis=0)

        nmse_pwd[n_s], nmse_pwd_cnn[n_s], nmse_pwd_pm[n_s] = \
            results_utils.nmse(P_pwd, P_gt), results_utils.nmse(P_pwd_cnn, P_gt), results_utils.nmse(P_pwd_pm, P_gt)
        ssim_pwd[n_s], ssim_pwd_cnn[n_s], ssim_pwd_pm[n_s] = \
            results_utils.ssim_freq(P_pwd, P_gt), results_utils.ssim_freq(P_pwd_cnn, P_gt), \
            results_utils.ssim_freq(P_pwd_pm, P_gt)

        if PLOT:
            # Plot params
            selection = np.ones_like(params_linear.array_pos[:, 0])
            selection[idx_missing] = 0
            selection = selection == 1
            n_f = 41
            print(str(params_linear.f_axis[n_f]))
            cmap = 'RdBu_r'
            tick_font_size = 70
            axis_label_size = 90

            # Ground truth
            plot_paths = os.path.join('plots', 'linear')
            save_path = os.path.join(plot_paths, 'sf_real_source_'
                                     + str(n_s) + '_f_' + str(params_linear.f_axis[n_f]) +'.pdf')
            results_utils.plot_soundfield(cmap, P_gt, n_f, selection, axis_label_size, tick_font_size,
                                          save_path, plot_ldspks=False, array_type='linear')

            # PWD
            save_path = os.path.join(plot_paths, 'sf_pwd_'
                                     + str(n_s) + '_f_' + str(params_linear.f_axis[n_f]) + '_n_l_' +str(args.n_missing) + '.pdf')
            results_utils.plot_soundfield(cmap, P_pwd, n_f, selection, axis_label_size, tick_font_size,
                                          save_path, array_type='linear')

            # Error
            nmse_pwd = 10 * np.log10(results_utils.nmse(P_pwd, P_gt, type='full'))
            save_path = os.path.join(plot_paths, 'nmse_pwd_'
                                     + str(n_s) + '_f_' + str(params_linear.f_axis[n_f]) + '_n_l_' +str(args.n_missing)+ '.pdf')
            results_utils.plot_soundfield(cmap, nmse_pwd, n_f, selection, axis_label_size, tick_font_size,
                                          save_path, do_norm=False, array_type='linear')

            # PWD-CNN
            save_path = os.path.join(plot_paths, 'sf_pwd_cnn_'
                                     + str(n_s) + '_f_' + str(params_linear.f_axis[n_f]) + '_n_l_' +str(args.n_missing)+ '.pdf')
            results_utils.plot_soundfield(cmap, P_pwd_cnn, n_f, selection, axis_label_size, tick_font_size,
                                          save_path, array_type='linear')

            # Error
            nmse_pwd_cnn = 10 * np.log10(results_utils.nmse(P_pwd_cnn, P_gt, type='full'))
            save_path = os.path.join(plot_paths, 'nmse_pwd_cnn_'
                                     + str(n_s) + '_f_' + str(params_linear.f_axis[n_f]) + '_n_l_' +str(args.n_missing)+ '.pdf')
            results_utils.plot_soundfield(cmap, nmse_pwd_cnn, n_f, selection, axis_label_size, tick_font_size,
                                          save_path, do_norm=False, array_type='linear')

            # PM
            save_path = os.path.join(plot_paths, 'sf_pm_'
                                     + str(n_s) + '_f_' + str(params_linear.f_axis[n_f]) + '_n_l_' +str(args.n_missing)+ '.pdf')
            results_utils.plot_soundfield(cmap, P_pwd_pm, n_f, selection, axis_label_size, tick_font_size,
                                          save_path, array_type='linear')

            # Error
            nmse_pm = 10 * np.log10(results_utils.nmse(P_pwd_pm, P_gt, type='full'))
            save_path = os.path.join(plot_paths, 'nmse_pm_'
                                     + str(n_s) + '_f_' + str(params_linear.f_axis[n_f]) + '_n_l_' +str(args.n_missing) + '.pdf')
            results_utils.plot_soundfield(cmap, nmse_pm, n_f, selection, axis_label_size, tick_font_size, save_path,
                                          do_norm=False, array_type='linear')

            print('bella')

    # If we are plotting it means we are just computing data for the paper --> no need to save anything
    if not PLOT:
        # Save arrays
        np.savez(os.path.join(dataset_path,'nmse_missing_'+str(args.n_missing)+'.npz'),
                 nmse_pwd=nmse_pwd, nmse_pwd_cnn=nmse_pwd_cnn, nmse_pwd_pm=nmse_pwd_pm)
        np.savez(os.path.join(dataset_path,'ssim_missing_'+str(args.n_missing)+'.npz'),
                 ssim_pwd=ssim_pwd, ssim_pwd_cnn=ssim_pwd_cnn, ssim_pwd_pm=ssim_pwd_pm)


if __name__ == '__main__':
    main()
