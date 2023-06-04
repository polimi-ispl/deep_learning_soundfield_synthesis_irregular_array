import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import sfs
import os

#import jax.numpy as jnp

def sound_field(d, selection, secondary_source, array, grid, tapering=False):
    if tapering:
        tapering_window = sfs.tapering.tukey(selection, alpha=0.3)
    else:
        tapering_window = sfs.tapering.none(selection)
    p = sfs.fd.synthesize(d, tapering_window, array, secondary_source, grid=grid)
    sfs.plot2d.amplitude(p, grid, xnorm=[0, 0, 0])
    sfs.plot2d.loudspeakers(array.x, array.n, tapering_window)

def main():
    # Arguments parse
    parser = argparse.ArgumentParser(description='Generate data for circular array setup')
    parser.add_argument('--dataset_path', type=str, help="Base Data Directory", default='/nas/home/lcomanducci/soundfield_synthesis/dataset/test/circular_array')
    parser.add_argument('--models_path', type=str, help='Deep learning models folder', default='/nas/home/lcomanducci/soundfield_synthesis/models/circular_array/')
    parser.add_argument('--gt_soundfield', type=bool, help='compute ground truth soundfield', default=True)
    parser.add_argument('--pm', type=bool, help='compute pressure matching', default=True)
    parser.add_argument('--pwd', type=bool, help='compute model-based acoustic rendering', default=True)
    parser.add_argument('--pwd_cnn', type=bool, help='compute model-based acoustic rendering + CNN', default=False)
    parser.add_argument('--pwd_cnn_complex', type=bool, help='compute model-based acoustic rendering + CNN complex', default=False)
    parser.add_argument('--wfs', type=bool, help='compute model-based acoustic rendering + CNN complex', default=False)
    parser.add_argument('--awfs', type=bool, help='compute model-based acoustic rendering + CNN complex', default=False)
    parser.add_argument('--pwd_apwd', type=bool, help='compute model-based acoustic rendering + adaptive wfs-like optimization', default=False)
    parser.add_argument('--n_missing', type=int, help='number missing loudspeakers',default=32)
    parser.add_argument('--gpu', type=str, help='number missing loudspeakers',default='0')

    PLOTPLOT = False
    PLOT = False

    if PLOT == False:
        eval_points = True
    else:
        eval_points = False

    args = parser.parse_args()
    dataset_path =args.dataset_path

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import scipy
    import tqdm
    import tensorflow as tf
    from data_lib import params_circular
    from data_lib import soundfield_generation as sg, results_utils

    # Grid of points where we actually compute the soundfield
    point = params_circular.point
    N_pts = len(point)     #params_circular.radius_sources_test

    # Load green function secondary sources --> eval points (it is in train directory since it is the same)
    dataset_path_train = '/nas/home/lcomanducci/soundfield_synthesis/dataset/circular_array'
    green_function_sec_sources_path = 'green_function_sec_sources_nl_' + str(params_circular.N_lspks) + '_r_' + str(params_circular.radius) + '.npy'
    G = np.load(os.path.join(dataset_path_train, green_function_sec_sources_path))

    # Load Missing loudspeakers configuration
    lspk_config_path = 'lspk_config_nl_' + str(params_circular.N_lspks) + '_missing_' + str(args.n_missing) + '.npy'
    lspk_config_path_global = os.path.join(dataset_path_train, 'setup', lspk_config_path)
    idx_missing = np.load(lspk_config_path_global)
    N_lspks = params_circular.N_lspks - args.n_missing
    theta_l = np.delete(params_circular.theta_l, idx_missing)
    G = np.delete(G, idx_missing, axis=1)

    # Let's precompute what we need in order to apply the selected models
    # Load pwd_cnn deep learning model
    N, h, theta_n, trunc_mod_exp_idx = sg.model_based_synthesis_circular_array(N_lspks, theta_l,params_circular.radius)

    if args.pwd_cnn_complex:
        model_name_complex = '/nas/home/lcomanducci/soundfield_synthesis/models/circular_array/model_circular_config_nl_64_missing_' +str(args.n_missing)+'_COMPLEX_LESS_CP_'+str(len(params_circular.point_cp))+'_lr1e-3'
        network_model_complex = tf.keras.models.load_model(os.path.join(args.models_path, model_name_complex))
        print(model_name_complex)


    if args.pm:
        G_cp = G[params_circular.idx_lr[params_circular.idx_cp]]  # Green function at control points

        reg_array = np.zeros(params_circular.N_freqs)
        for n_f in range(params_circular.N_freqs):
            _, s, _ = np.linalg.svd(np.matmul(G_cp[:, :, n_f].transpose(), G_cp[:, :, n_f]))
            reg_array[n_f] = np.max(s)*1e-3

        G_cp = G[params_circular.idx_lr[params_circular.idx_cp]]  # Green function at control points
        C_pm = np.zeros((N_lspks, len(params_circular.idx_cp), params_circular.N_freqs ),dtype=complex)
        for n_f in tqdm.tqdm(range(len(params_circular.wc))):
            C_pm[:, :, n_f] = np.matmul(np.linalg.pinv(np.matmul(G_cp[:, :, n_f].transpose(), G_cp[:, :, n_f]) + reg_array[n_f] * np.eye(N_lspks)),G_cp[:, :, n_f].transpose())


    if eval_points:
        N_pts = len(params_circular.idx_lr)
        G = G[params_circular.idx_lr]
        point = params_circular.point_lr

    N_src_radius = len(params_circular.radius_sources_train)
    #N_src_radius = 1
    nmse_pwd = np.zeros((N_src_radius, params_circular.n_sources_radius_test, params_circular.N_freqs), dtype=complex)
    nmse_pwd_cnn = np.zeros_like(nmse_pwd, dtype=complex)
    nmse_pwd_pm = np.zeros_like(nmse_pwd, dtype=complex)
    nmse_pwd_cnn_complex = np.zeros_like(nmse_pwd, dtype=complex)
    nmse_pwd_apwd = np.zeros_like(nmse_pwd, dtype=complex)
    nmse_wfs = np.zeros_like(nmse_pwd, dtype=complex)
    nmse_awfs = np.zeros_like(nmse_pwd, dtype=complex)

    ssim_pwd = np.zeros_like(nmse_pwd, dtype=complex)
    ssim_pwd_cnn = np.zeros_like(nmse_pwd, dtype=complex)
    ssim_pwd_pm = np.zeros_like(nmse_pwd, dtype=complex)
    ssim_pwd_cnn_complex = np.zeros_like(nmse_pwd, dtype=complex)
    ssim_pwd_apwd = np.zeros_like(nmse_pwd, dtype=complex)
    ssim_wfs = np.zeros_like(nmse_pwd, dtype=complex)
    ssim_awfs = np.zeros_like(nmse_pwd, dtype=complex)

    for n_r in tqdm.tqdm(range(N_src_radius)):
        P_gt = np.zeros((N_pts, params_circular.N_freqs), dtype=complex)
        P_pwd = np.zeros((N_pts, params_circular.N_freqs), dtype=complex)
        P_pwd_cnn = np.zeros((N_pts, params_circular.N_freqs), dtype=complex)
        P_pwd_pm = np.zeros((N_pts, params_circular.N_freqs), dtype=complex)
        P_pwd_cnn_complex = np.zeros((N_pts, params_circular.N_freqs), dtype=complex)
        P_pwd_apwd = np.zeros((N_pts, params_circular.N_freqs), dtype=complex)
        P_wfs = np.zeros((N_pts, params_circular.N_freqs), dtype=complex)
        P_awfs  = np.zeros((N_pts, params_circular.N_freqs), dtype=complex)

        for n_s in tqdm.tqdm(range(params_circular.n_sources_radius_test)):
            if PLOT:
                n_s = 22
            if PLOTPLOT:
                n_s = np.random.randint(0,params_circular.n_sources_radius_test)
            xs = params_circular.src_pos_test[n_r, n_s]

            if args.gt_soundfield:
                for n_f in range(params_circular.N_freqs):
                    hankel_arg = (params_circular.wc[n_f] / params_circular.c) * \
                                 np.linalg.norm(point[:, :2] - xs, axis=1)
                    P_gt[ :, n_f] = (1j / 4) * scipy.special.hankel1(0, hankel_arg)

            if args.pwd:
                Phi = sg.herglotz_density_point_source(xs, theta_n, trunc_mod_exp_idx, N, A=1)
                # Multiply filters for herglotz density function
                d_array = np.zeros((N_lspks, params_circular.N_freqs), dtype=complex)
                for n_f in range(params_circular.N_freqs):
                    d_array[ :, n_f] = (1 / N[n_f]) * np.matmul(np.expand_dims(Phi[n_f], axis=0), h[n_f].transpose())

                for n_p in range(N_pts):
                    P_pwd[n_p, :] = np.sum(G[n_p] * d_array, axis=0)

                if args.pwd_apwd:
                    # ADD ARG STUFF
                    # AWFS adapted
                    if eval_points:
                        e_pwd = P_gt[params_circular.idx_cp] - P_pwd[params_circular.idx_cp]
                    else:
                        e_pwd = P_gt[params_circular.idx_lr[params_circular.idx_cp]] - P_pwd[params_circular.idx_lr[params_circular.idx_cp]]

                    d_array_opt = np.zeros_like(d_array,dtype=complex)
                    for n_f in range(params_circular.N_freqs):
                        d_array_opt[:,n_f] = np.matmul(np.linalg.inv(np.matmul(np.conj(G_cp[:,:,n_f].T),G_cp[:,:,n_f]) + reg_array[n_f]*np.eye(N_lspks)), np.matmul(np.conj(G_cp[:,:,n_f].T),e_pwd[:,n_f]))+ d_array[:,n_f]

                    for n_p in range(N_pts):
                        P_pwd_apwd[n_p, :] = np.sum(G[n_p] * d_array_opt, axis=0)

            if args.wfs:
                # Simulation parameters
                d_array_wfs = np.zeros((N_lspks, params_circular.N_freqs), dtype=complex)
                for n_f in range(params_circular.N_freqs):
                    frequency =  params_circular.f_axis[n_f]# in Hz
                    #grid = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.02)
                    omega = 2 * np.pi * frequency  # angular frequency
                    array_wfs = sfs.array.as_secondary_source_distribution([np.delete(params_circular.array.x,idx_missing,axis=0), np.delete(params_circular.array.n,idx_missing,axis=0), np.delete(params_circular.array.a,idx_missing,axis=0)])
                    d, selection, secondary_source = sfs.fd.wfs.point_2d(omega, array_wfs.x, array_wfs.n, [xs[0],xs[1],0])
                    d_array_wfs[:,n_f] = d

                for n_p in range(N_pts):
                    P_wfs[n_p, :] = np.sum(G[n_p] * d_array_wfs, axis=0)

            if args.awfs:
                if eval_points:
                    e_pwfs = P_gt[params_circular.idx_cp] - P_wfs[params_circular.idx_cp]
                else:
                    e_pwfs = P_gt[params_circular.idx_lr[params_circular.idx_cp]] - P_wfs[
                        params_circular.idx_lr[params_circular.idx_cp]]

                d_array_awfs = np.zeros_like(d_array_wfs, dtype=complex)
                for n_f in range(params_circular.N_freqs):
                    d_array_awfs[:, n_f] = np.matmul(np.linalg.inv(
                        np.matmul(np.conj(G_cp[:, :, n_f].T), G_cp[:, :, n_f]) + reg_array[n_f] * np.eye(N_lspks)),
                                                    np.matmul(np.conj(G_cp[:, :, n_f].T), e_pwfs[:, n_f])) + d_array_wfs[:,
                                                                                                            n_f]
                for n_p in range(N_pts):
                    P_awfs[n_p, :] = np.sum(G[n_p] * d_array_awfs, axis=0)


            if args.pwd_cnn_complex:
                d_array_cnn_complex = network_model_complex.predict(np.expand_dims(d_array, axis=[0, -1]),verbose=0)[0, :, :, 0]
                for n_p in range(N_pts):
                    P_pwd_cnn_complex[n_p, :] = np.sum(G[n_p] * d_array_cnn_complex, axis=0)

            if args.pm:
                d_pm = np.zeros((N_lspks, params_circular.N_freqs),dtype=complex)
                for n_f in range(params_circular.N_freqs):
                    if eval_points:
                        d_pm[:, n_f] = np.matmul(C_pm[:, :, n_f], P_gt[params_circular.idx_cp, n_f]) # CHECK idx_cp is USING EVAL POINTSSSSS
                    else:
                        d_pm[:, n_f] = np.matmul(C_pm[:, :, n_f], P_gt[params_circular.idx_lr[params_circular.idx_cp], n_f]) # CHECK idx_cp is USING EVAL POINTSSSSS

                for n_p in range(N_pts):
                    P_pwd_pm[n_p, :] = np.sum(G[n_p] * d_pm, axis=0)

            nmse_pwd[n_r, n_s], nmse_pwd_cnn[n_r, n_s], nmse_pwd_pm[n_r, n_s] = results_utils.nmse(P_pwd, P_gt), results_utils.nmse(P_pwd_cnn, P_gt), results_utils.nmse(P_pwd_pm, P_gt)
            ssim_pwd[n_r, n_s], ssim_pwd_cnn[n_r, n_s], ssim_pwd_pm[n_r, n_s] = results_utils.ssim_freq(P_pwd, P_gt), results_utils.ssim_freq(P_pwd_cnn, P_gt), results_utils.ssim_freq(P_pwd_pm, P_gt)
            nmse_pwd_cnn_complex[n_r, n_s] = results_utils.nmse(P_pwd_cnn_complex, P_gt)
            ssim_pwd_cnn_complex[n_r, n_s] = results_utils.ssim_freq(P_pwd_cnn_complex, P_gt)

            ssim_wfs[n_r, n_s] = results_utils.ssim_freq(P_wfs, P_gt)
            nmse_wfs[n_r, n_s] =  results_utils.nmse(P_wfs, P_gt)
            ssim_awfs[n_r, n_s] = results_utils.ssim_freq(P_awfs, P_gt)
            nmse_awfs[n_r, n_s] =  results_utils.nmse(P_awfs, P_gt)
            nmse_pwd_apwd[n_r, n_s] = results_utils.nmse(P_pwd_apwd, P_gt)
            ssim_pwd_apwd[n_r, n_s] = results_utils.ssim_freq(P_pwd_apwd, P_gt)



            if PLOTPLOT:
                plt.figure(figsize=(10, 10))
                n_f = 41
                plt.subplot(4, 2, 1)
                plt.imshow(np.reshape(np.real(P_gt[:, n_f]), (200, 200)), aspect='auto'), plt.colorbar(), plt.title('gt')
                plt.tight_layout()
                plt.subplot(4, 2, 2)
                plt.imshow(np.reshape(np.real(P_pwd[:, n_f]), (200, 200)), aspect='auto'), plt.colorbar(), plt.title('pwd')
                plt.tight_layout()
                plt.subplot(4, 2, 3)
                plt.imshow(np.reshape(np.real(P_pwd_apwd[:, n_f]), (200, 200)), aspect='auto'), plt.colorbar(), plt.title(
                    'apwd')
                plt.tight_layout()
                plt.subplot(4, 2, 4)
                plt.imshow(np.reshape(np.real(P_pwd_cnn_complex[:, n_f]), (200, 200)),
                           aspect='auto'), plt.colorbar(), plt.title('cnn_cpx')
                plt.tight_layout()
                plt.subplot(4, 2, 5)
                plt.imshow(np.reshape(np.real(P_wfs[:, n_f]), (200, 200)), aspect='auto'), plt.colorbar(), plt.title('wfs')
                plt.tight_layout()
                plt.subplot(4, 2, 6)
                plt.imshow(np.reshape(np.real(P_awfs[:, n_f]), (200, 200)), aspect='auto'), plt.colorbar(), plt.title(
                    'awfs')
                plt.tight_layout()
                plt.subplot(4, 2, 7)
                plt.imshow(np.reshape(np.real(P_pwd_pm[:, n_f]), (200, 200)), aspect='auto'), plt.colorbar(), plt.title(
                    'pm')
                plt.tight_layout()
                plt.show()

                nmse_pwd = 10 * np.log10(results_utils.nmse(P_pwd, P_gt, type='full'))
                nmse_pwd_apwd = 10 * np.log10(results_utils.nmse(P_pwd_apwd, P_gt, type='full'))
                nmse_pwd_cnn_complex = 10 * np.log10(results_utils.nmse(P_pwd_cnn_complex, P_gt, type='full'))
                nmse_wfs = 10 * np.log10(results_utils.nmse(P_wfs, P_gt, type='full'))
                nmse_awfs = 10 * np.log10(results_utils.nmse(P_awfs, P_gt, type='full'))
                nmse_pm = 10 * np.log10(results_utils.nmse(P_pwd_pm, P_gt, type='full'))

                plt.figure()
                plt.plot(10 * np.log10(results_utils.nmse(P_pwd_cnn_complex, P_gt)))
                plt.plot(10 * np.log10(results_utils.nmse(P_pwd_apwd, P_gt)))
                plt.legend(['complex', 'apwd'])
                plt.show()

                plt.figure(figsize=(10, 10))
                n_f = 41
                plt.subplot(4, 2, 1)
                plt.imshow(np.reshape(np.real(nmse_pwd[:, n_f]), (200, 200)), aspect='auto'), plt.colorbar(), plt.title(
                    'pwd')
                plt.tight_layout()
                plt.subplot(4, 2, 2)
                plt.imshow(np.reshape(np.real(nmse_pwd_apwd[:, n_f]), (200, 200)),
                           aspect='auto'), plt.colorbar(), plt.title('apwd')
                plt.tight_layout()
                plt.subplot(4, 2, 3)
                plt.imshow(np.reshape(np.real(nmse_pwd_cnn_complex[:, n_f]), (200, 200)),
                           aspect='auto'), plt.colorbar(), plt.title(
                    'coomplex')
                plt.tight_layout()
                plt.subplot(4, 2, 4)
                plt.imshow(np.reshape(np.real(nmse_wfs[:, n_f]), (200, 200)),
                           aspect='auto'), plt.colorbar(), plt.title('wfs')
                plt.tight_layout()
                plt.subplot(4, 2, 5)
                plt.imshow(np.reshape(np.real(nmse_awfs[:, n_f]), (200, 200)), aspect='auto'), plt.colorbar(), plt.title(
                    'awfs')
                plt.tight_layout()
                plt.subplot(4, 2, 6)
                plt.imshow(np.reshape(np.real(nmse_pm[:, n_f]), (200, 200)), aspect='auto'), plt.colorbar(), plt.title(
                    'pm')
                plt.tight_layout()
                plt.show()

            if PLOT:
                # Plot params
                selection = np.ones_like(params_circular.array_pos[:, 0])
                selection[idx_missing] = 0
                selection = selection == 1
                n_s = 32
                n_f = 41
                print(str(params_circular.f_axis[n_f]))
                cmap = 'RdBu_r'
                tick_font_size = 70
                axis_label_size = 90

                # Ground truth
                plot_paths = os.path.join('plots', 'circular')
                save_path = os.path.join(plot_paths, 'sf_real_source_'
                                         + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '.pdf')
                results_utils.plot_soundfield(cmap, P_gt, n_f, selection, axis_label_size, tick_font_size, save_path, plot_ldspks=False, array_type='circular')

                if args.pwd:
                    # PWD
                    save_path = os.path.join(plot_paths, 'sf_pwd_'
                                             + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '.pdf')
                    results_utils.plot_soundfield(cmap, P_pwd, n_f, selection, axis_label_size, tick_font_size, save_path, array_type='circular')

                    # Error

                    nmse_pwd = 10*np.log10(results_utils.nmse(P_pwd, P_gt, type='full'))
                    save_path = os.path.join(plot_paths, 'nmse_pwd_'
                                             + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '.pdf')
                    results_utils.plot_soundfield(cmap, nmse_pwd, n_f, selection, axis_label_size, tick_font_size, save_path, do_norm=False, array_type='circular')
                if args.pwd_apwd:
                    # PWD AWFS
                    save_path = os.path.join(plot_paths, 'sf_pwd_apwd'
                                             + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '.pdf')
                    results_utils.plot_soundfield(cmap, P_pwd_apwd, n_f, selection, axis_label_size, tick_font_size, save_path,
                                                  array_type='circular')

                    # Error
                    nmse_pwd_apwd = 10 * np.log10(results_utils.nmse(P_pwd_apwd, P_gt, type='full'))
                    save_path = os.path.join(plot_paths, 'nmse_pwd_apwd'
                                             + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '.pdf')
                    results_utils.plot_soundfield(cmap, nmse_pwd_apwd, n_f, selection, axis_label_size, tick_font_size,
                                                  save_path, do_norm=False, array_type='circular')
                if args.pwd_cnn:
                    # PWD-CNN
                    save_path = os.path.join(plot_paths, 'sf_pwd_cnn_'
                                             + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '.pdf')
                    results_utils.plot_soundfield(cmap, P_pwd_cnn, n_f, selection, axis_label_size, tick_font_size, save_path, array_type='circular')

                    # Error
                    nmse_pwd_cnn = 10*np.log10(results_utils.nmse(P_pwd_cnn, P_gt, type='full'))
                    save_path = os.path.join(plot_paths, 'nmse_pwd_cnn_'
                                             + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '.pdf')
                    results_utils.plot_soundfield(cmap, nmse_pwd_cnn, n_f, selection, axis_label_size, tick_font_size, save_path, do_norm=False, array_type='circular')

                if args.pwd_cnn_complex:
                    # PWD-CNN-COMPLEX
                    save_path = os.path.join(plot_paths, 'sf_pwd_cnn_'
                                             + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '.pdf')
                    results_utils.plot_soundfield(cmap, P_pwd_cnn_complex, n_f, selection, axis_label_size, tick_font_size, save_path, array_type='circular')

                    # Error
                    nmse_pwd_cnn_complex = 10*np.log10(results_utils.nmse(P_pwd_cnn_complex, P_gt, type='full'))
                    save_path = os.path.join(plot_paths, 'nmse_pwd_cnn_complex'
                                             + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '.pdf')
                    results_utils.plot_soundfield(cmap, nmse_pwd_cnn_complex, n_f, selection, axis_label_size, tick_font_size, save_path, do_norm=False, array_type='circular')

                if args.pm:
                    # PM
                    save_path = os.path.join(plot_paths, 'sf_pm_'
                                             + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '.pdf')
                    results_utils.plot_soundfield(cmap, P_pwd_pm, n_f, selection, axis_label_size, tick_font_size, save_path, array_type='circular')

                    # Error
                    nmse_pm = 10*np.log10(results_utils.nmse(P_pwd_pm, P_gt, type='full'))
                    save_path = os.path.join(plot_paths, 'nmse_pm_'
                                             + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '.pdf')
                    results_utils.plot_soundfield(cmap, nmse_pm, n_f, selection, axis_label_size, tick_font_size, save_path, do_norm=False, array_type='circular')

                if args.wfs:
                    # DLPM
                    save_path = os.path.join(plot_paths, 'sf_wfs_'
                                             + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '.pdf')
                    results_utils.plot_soundfield(cmap, P_wfs, n_f, selection, axis_label_size, tick_font_size, save_path, array_type='circular')

                    # Error
                    nmse_wfs = 10*np.log10(results_utils.nmse(P_wfs, P_gt, type='full'))
                    save_path = os.path.join(plot_paths, 'nmse_wfs_'
                                             + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '.pdf')

                    results_utils.plot_soundfield(cmap, nmse_wfs, n_f, selection, axis_label_size, tick_font_size, save_path, do_norm=False, array_type='circular')

                if args.awfs:
                    # DLPM
                    save_path = os.path.join(plot_paths, 'sf_awfs_'
                                             + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '.pdf')
                    results_utils.plot_soundfield(cmap, P_awfs, n_f, selection, axis_label_size, tick_font_size, save_path, array_type='circular')

                    # Error
                    nmse_awfs = 10*np.log10(results_utils.nmse(P_awfs, P_gt, type='full'))
                    save_path = os.path.join(plot_paths, 'nmse_awfs_'
                                             + str(n_s) + '_f_' + str(params_circular.f_axis[n_f]) + '.pdf')

        plt.figure()
        plt.plot(10*np.log10(np.mean(np.real(np.reshape(nmse_pwd,(-1,params_circular.N_freqs))), axis=0)))

        plt.plot(10*np.log10(np.mean(np.real(nmse_pwd_cnn_complex[0]), axis=0)))
        # plt.plot(np.mean(np.real(nmse_awfs[0]),axis=0))
        plt.plot(10*np.log10(np.mean(np.real(nmse_pwd_apwd[0]), axis=0)))
        plt.legend(['pwd', 'pwd_cpx', 'apwd'])
        plt.show()
        print('next radius')
    print('cycle done')
    # If we are plotting it means we are just computing data for the paper --> no need to save anything
    if not PLOT:
        # Save arrays
        np.savez(os.path.join(dataset_path, 'nmse_missing_'+str(args.n_missing)+'.npz'),
                 nmse_pwd=nmse_pwd,  nmse_pwd_pm=nmse_pwd_pm,
                 nmse_pwd_cnn_complex=nmse_pwd_cnn_complex,nmse_pwd_apwd=nmse_pwd_apwd, nmse_wfs=nmse_wfs, nmse_awfs=nmse_awfs)
        np.savez(os.path.join(dataset_path, 'ssim_missing_'+str(args.n_missing)+'.npz'),
                 ssim_pwd=ssim_pwd,  ssim_pwd_pm=ssim_pwd_pm,
                 ssim_pwd_cnn_complex=ssim_pwd_cnn_complex,ssim_pwd_apwd=ssim_pwd_apwd, ssim_wfs=ssim_wfs, ssim_awfs=ssim_awfs)

if __name__ == '__main__':
    main()
