import numpy as np
import scipy
import tqdm
import argparse
import tensorflow as tf
import sfs
import time
import os



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def main():
    # Arguments parse
    parser = argparse.ArgumentParser(description='Generate data for circular array setup')
    parser.add_argument('--dataset_path', type=str, help="Base Data Directory", default='/nas/home/lcomanducci/soundfield_synthesis/dataset/test/circular_array')

    parser.add_argument('--models_path', type=str, help='Deep learning models folder',
                        default='/nas/home/lcomanducci/soundfield_synthesis_RQ/models/circular_array/')
    parser.add_argument('--gt_soundfield', type=bool, help='compute ground truth soundfield', default=True)
    parser.add_argument('--pm', type=bool, help='compute pressure matching', default=False)
    parser.add_argument('--pwd', type=bool, help='compute model-based acoustic rendering', default=False)
    parser.add_argument('--pwd_cnn', type=bool, help='compute model-based acoustic rendering + CNN', default=False)
    parser.add_argument('--n_missing', type=int, help='number missing loudspeakers', default=16)
    parser.add_argument('--wfs', type=bool, help='compute Wave Field Synthesis', default=False)
    parser.add_argument('--awfs', type=bool, help='compute Adaptive Wave Field Synthesis', default=False)
    parser.add_argument('--pwd_apwd', type=bool,
                        help='compute Adaptive model-based acoustic rendering + adaptive wfs-like optimization',
                        default=False)
    parser.add_argument('--gpu', type=str, help='GPU NUMBER', default='0')
    parser.add_argument('--use_jax', type=str, help='leverage jax to speed up computation time', default='False')

    # Load missing packages here to be able to select GPU
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import tensorflow as tf
    if args.use_jax == True:
        import jax.numpy as jnp
    from data_lib import params_circular
    from data_lib import soundfield_generation as sg, results_utils

    eval_points = True
    args = parser.parse_args()
    dataset_path = args.dataset_path

    # Grid of points where we actually compute the soundfield
    point = params_circular.point
    N_pts = len(point)
    # Load green function secondary sources --> eval points (it is in train directory since it is the same)
    dataset_path_train = '/nas/home/lcomanducci/soundfield_synthesis/dataset/circular_array'
    green_function_sec_sources_path = 'green_function_sec_sources_nl_' + str(params_circular.N_lspks) + '_r_' + str(params_circular.radius) + '.npy'
    G = np.load(os.path.join(dataset_path_train, green_function_sec_sources_path)).astype(np.complex64)

    # Load Missing loudspeakers configuration
    lspk_config_path = 'lspk_config_nl_' + str(params_circular.N_lspks) + '_missing_' + str(args.n_missing) + '.npy'
    lspk_config_path_global = os.path.join(dataset_path_train, 'setup', lspk_config_path)
    idx_missing = np.load(lspk_config_path_global)
    N_lspks = params_circular.N_lspks - args.n_missing
    theta_l = np.delete(params_circular.theta_l, idx_missing)
    G = np.delete(G, idx_missing, axis=1)

    array_wfs = sfs.array.as_secondary_source_distribution(
        [np.delete(params_circular.array.x, idx_missing, axis=0),
         np.delete(params_circular.array.n, idx_missing, axis=0),
         np.delete(params_circular.array.a, idx_missing, axis=0)])

    # Let's precompute what we need in order to apply the selected models
    # Precompute for pwd and cnn
    N, h, theta_n, trunc_mod_exp_idx = sg.model_based_synthesis_circular_array(N_lspks, theta_l)

    # Regularization params as in Ueno, N., Koyama, S., & Saruwatari, H. (2019). Three-dimensional sound field reproduction based on weighted mode-matching method. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 27(12), 1852-1867.
    G_cp = G[params_circular.idx_lr[params_circular.idx_cp]]  # Green function at control points
    reg_array = np.zeros(params_circular.N_freqs)
    for n_f in range(params_circular.N_freqs):
        _, s, _ = np.linalg.svd(np.matmul(np.conj(G_cp[:, :, n_f].transpose()), G_cp[:, :, n_f]))
        reg_array[n_f] = np.max(s) * 1e-3

    # Load pwd_cnn deep learning model
    if args.pwd_cnn:
        model_name = '/nas/home/lcomanducci/soundfield_synthesis_RQ/models/circular_array/model_circular_config_nl_64_missing_' \
                     + str(args.n_missing) + '_COMPLEX_CP_' + str(
            len(params_circular.point_cp)) + '_lr_0.0001PReLU_earlyStop_10'
        print('Loaded model '+model_name)
        network_model = tf.keras.models.load_model(os.path.join(args.models_path, model_name))

        # Driving signals - Model-based acoustic rendering

    if args.pm:
        C_pm = np.zeros((N_lspks, len(params_circular.idx_cp), params_circular.N_freqs), dtype=complex)
        for n_f in tqdm.tqdm(range(len(params_circular.wc))):
            C_pm[:, :, n_f] = np.matmul(np.linalg.pinv(
                np.matmul(G_cp[:, :, n_f].transpose(), G_cp[:, :, n_f]) + reg_array[n_f] * np.eye(N_lspks)),
                                        G_cp[:, :, n_f].transpose())

    N_pts = len(params_circular.idx_lr)
    G = G[params_circular.idx_lr]
    point = params_circular.point_lr

    N_radius = len(params_circular.radius_sources_train)
    N_sources_radius = params_circular.n_sources_radius

    nmse_pwd = np.zeros((N_radius, N_sources_radius, params_circular.N_freqs), dtype=complex)
    nmse_pwd_cnn, nmse_pwd_pm = np.zeros_like(nmse_pwd), np.zeros_like(nmse_pwd)
    nmse_pwd_apwd = np.zeros_like(nmse_pwd, dtype=np.complex64)
    nmse_wfs = np.zeros_like(nmse_pwd, dtype=np.complex64)
    nmse_awfs = np.zeros_like(nmse_pwd, dtype=np.complex64)
    ssim_pwd, ssim_pwd_cnn, ssim_pwd_pm = np.zeros_like(nmse_pwd), np.zeros_like(nmse_pwd), np.zeros_like(nmse_pwd)
    ssim_pwd_apwd = np.zeros_like(nmse_pwd, dtype=np.complex64)
    ssim_wfs = np.zeros_like(nmse_pwd, dtype=np.complex64)
    ssim_awfs = np.zeros_like(nmse_pwd, dtype=np.complex64)
    P_gt = np.zeros((N_radius,N_sources_radius, N_pts, params_circular.N_freqs), dtype=np.complex64)


    print('COMPUTE GROUND TRUTH SOUNDFIELD')
    if args.gt_soundfield:
        path_soundfield = os.path.join(dataset_path, 'gt_soundfield_test' + '_cp' + str(len(params_circular.idx_cp)) + '.npy')
        if os.path.exists(path_soundfield):
            tic = time.time()
            P_gt = np.load(path_soundfield)
            print(str(time.time()-tic)+'s for loading gt soundfield....')
        else:
            for n_r in tqdm.tqdm(range(len(params_circular.radius_sources_train))):
                for n_s in tqdm.tqdm(range(params_circular.n_sources_radius)):

                    xs = params_circular.src_pos_test[n_r, n_s]
                    for n_f in range(params_circular.N_freqs):
                        hankel_arg = (params_circular.wc[n_f] / params_circular.c) * \
                                     np.linalg.norm(point[:, :2] - xs, axis=1)
                        P_gt[n_r,n_s, :, n_f] = (1j / 4) * scipy.special.hankel2(0, hankel_arg)
            np.save(path_soundfield,P_gt)


    path_pwd = os.path.join(dataset_path, 'filters_test_config_nl_' + str(params_circular.N_lspks) + '_missing_' + str(
        args.n_missing) + '_cp' + str(len(params_circular.idx_cp)) + '.npy')
    if args.pwd:
        print('COMPUTE PWD')  ###########################################################################################
        if os.path.exists(path_pwd):
            d_array = np.load(path_pwd)
        else:
            d_array = np.zeros((N_radius, N_sources_radius, N_lspks, params_circular.N_freqs), dtype=np.complex64)
            for n_r in tqdm.tqdm(range(N_radius)):
                for n_s in tqdm.tqdm(range(N_sources_radius)):

                    xs = params_circular.src_pos_test[n_r, n_s]
                    Phi = sg.herglotz_density_point_source(xs, theta_n, trunc_mod_exp_idx, N, A=1)

                    # Multiply filters for herglotz density function
                    for n_f in range(params_circular.N_freqs):
                        d_array[n_r, n_s, :, n_f] = (1 / N[n_f]) * np.matmul(np.expand_dims(Phi[n_f], axis=0), h[n_f].transpose())
            np.save(path_pwd,d_array)

        for n_r in tqdm.tqdm(range(N_radius)):
            for n_s in tqdm.tqdm(range(N_sources_radius)):
                # PWD
                P_pwd=np.einsum('ijk,jk->ik', G, d_array[n_r,n_s])
                nmse_pwd[n_r,n_s] = results_utils.nmse(P_pwd, P_gt[n_r,n_s])
                ssim_pwd[n_r,n_s] = results_utils.ssim_freq(P_pwd, P_gt[n_r,n_s])

                # APWD
                if args.pwd_apwd:
                    e_pwd = P_gt[n_r, n_s, params_circular.idx_cp] - P_pwd[params_circular.idx_cp]
                    d_array_opt = np.zeros_like(d_array[n_r,n_s], dtype=np.complex64)
                    for n_f in range(params_circular.N_freqs):
                        d_array_opt[:, n_f] = np.matmul(np.linalg.inv(
                            np.matmul(np.conj(G_cp[:, :, n_f].T), G_cp[:, :, n_f]) + reg_array[n_f] * np.eye(N_lspks)),
                            np.matmul(np.conj(G_cp[:, :, n_f].T), e_pwd[:, n_f])) + d_array[n_r, n_s,:, n_f]
                    P_pwd_apwd = np.einsum('ijk,jk->ik', G, d_array_opt)
                    nmse_pwd_apwd[n_r,n_s], ssim_pwd_apwd[n_r,n_s] = results_utils.nmse(P_pwd_apwd, P_gt[n_r, n_s]), results_utils.ssim_freq(P_pwd_apwd, P_gt[n_r, n_s])
        np.save(os.path.join(dataset_path, 'nmse_pwd_missing_' + str(args.n_missing) + '.npy'),nmse_pwd)
        np.save(os.path.join(dataset_path, 'ssim_pwd_missing_' + str(args.n_missing) + '.npy'),ssim_pwd)
        np.save(os.path.join(dataset_path, 'nmse_pwd_apwd_missing_' + str(args.n_missing) + '.npy'),nmse_pwd_apwd)
        np.save(os.path.join(dataset_path, 'ssim_pwd_apwd_missing_' + str(args.n_missing) + '.npy'),ssim_pwd_apwd)


    if args.wfs:
        print('COMPUTE WFS')  #########################################################################################

        for n_r in tqdm.tqdm(range(N_radius)):
            for n_s in tqdm.tqdm(range(N_sources_radius)):
                xs = params_circular.src_pos_test[n_r, n_s]
                ### WFS
                d_array_wfs = np.zeros((N_lspks, params_circular.N_freqs), dtype=np.complex64)
                for n_f in range(params_circular.N_freqs):
                    frequency = params_circular.f_axis[n_f]  # in Hz
                    # grid = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.02)
                    omega = 2 * np.pi * frequency  # angular frequency
                    d, selection, secondary_source = sfs.fd.wfs.line_2d(omega, array_wfs.x, array_wfs.n,[xs[0], xs[1], 0])
                    d_array_wfs[:, n_f] = d
                P_wfs = np.einsum('ijk,jk->ik', G[:,selection], d_array_wfs[selection])

                nmse_wfs[n_r, n_s], ssim_wfs[n_r, n_s] = results_utils.nmse(P_wfs, P_gt[n_r, n_s]), results_utils.ssim_freq(P_wfs, P_gt[n_r, n_s])

                ### AWFS
                if args.awfs:
                    e_pwfs = P_gt[n_r, n_s,params_circular.idx_cp] - P_wfs[params_circular.idx_cp]
                    d_array_awfs = np.zeros_like(d_array_wfs[selection], dtype=complex)
                    for n_f in range(params_circular.N_freqs):
                        d_array_awfs[:, n_f] = np.matmul(np.linalg.inv(
                            np.matmul(np.conj(G_cp[:, selection, n_f].T), G_cp[:, selection, n_f]) + reg_array[
                                n_f] * np.eye(np.sum(selection))),
                            np.matmul(np.conj(G_cp[:, selection, n_f].T), e_pwfs[:, n_f])) + d_array_wfs[selection, n_f]
                    P_awfs = np.einsum('ijk,jk->ik', G[:,selection], d_array_awfs)
                    nmse_awfs[n_r, n_s], ssim_awfs[n_r, n_s] = results_utils.nmse(P_awfs, P_gt[n_r, n_s]), results_utils.ssim_freq(P_awfs, P_gt[n_r, n_s])

        np.save(os.path.join(dataset_path, 'nmse_awfs_missing_' + str(args.n_missing) + '.npy'), nmse_awfs)
        np.save(os.path.join(dataset_path, 'ssim_awfs_missing_' + str(args.n_missing) + '.npy'), ssim_awfs)
        np.save(os.path.join(dataset_path, 'nmse_wfs_missing_' + str(args.n_missing) + '.npy'), nmse_wfs)
        np.save(os.path.join(dataset_path, 'ssim_wfs_missing_' + str(args.n_missing) + '.npy'), ssim_wfs)

    if args.pm:
        print('COMPUTE PM')  ###########################################################################################
        print(P_gt.shape)
        d_pm = np.zeros((N_lspks, params_circular.N_freqs), dtype=np.complex64)
        for n_r in tqdm.tqdm(range(N_radius)):
            for n_s in tqdm.tqdm(range(N_sources_radius)):
                xs = params_circular.src_pos_test[n_r, n_s]

                for n_f in range(params_circular.N_freqs):
                    d_pm[:, n_f] = np.matmul(C_pm[:, :, n_f], P_gt[n_r,n_s, params_circular.idx_cp, n_f])
                P_pwd_pm = np.einsum('ijk,jk->ik', G, d_pm)
                nmse_pwd_pm[n_r, n_s], ssim_pwd_pm[n_r, n_s] = results_utils.nmse(P_pwd_pm, P_gt[n_r, n_s]), results_utils.ssim_freq(P_pwd_pm, P_gt[n_r, n_s])
        np.save(os.path.join(dataset_path, 'nmse_pwd_pm_missing_' + str(args.n_missing) + '.npy'), nmse_pwd_pm)
        np.save(os.path.join(dataset_path, 'ssim_pwd_pm_missing_' + str(args.n_missing) + '.npy'), ssim_pwd_pm)

    """"""
    if args.pwd_cnn:
        print('COMPUTE PWD_CNN')  ###########################################################################################
        d_array = np.load(path_pwd)
        d_array = tf.reshape(d_array,(N_radius*N_sources_radius,N_lspks,params_circular.N_freqs))
        d_array_cnn_complex = network_model.predict(np.expand_dims(d_array, axis=[-1]), verbose=0,batch_size=32)[:, :, :, 0]
        d_array_cnn_complex = tf.reshape(d_array_cnn_complex,(N_radius, N_sources_radius,N_lspks,params_circular.N_freqs))
        for n_r in tqdm.tqdm(range(N_radius)):
            for n_s in tqdm.tqdm(range(N_sources_radius)):
                P_pwd_cnn = np.einsum('ijk,jk->ik', G, d_array_cnn_complex[n_r, n_s])
                nmse_pwd_cnn[n_r, n_s], ssim_pwd_cnn[n_r, n_s] = results_utils.nmse(P_pwd_cnn, P_gt[n_r, n_s]), results_utils.ssim_freq(P_pwd_cnn, P_gt[n_r, n_s])
        np.save(os.path.join(dataset_path, 'nmse_pwd_cnn_missing_' + str(args.n_missing) + '.npy'), nmse_pwd_cnn)
        np.save(os.path.join(dataset_path, 'ssim_pwd_cnn_missing_' + str(args.n_missing) + '.npy'), ssim_pwd_cnn)

    if not args.pwd:
        nmse_pwd = np.load(os.path.join(dataset_path, 'nmse_pwd_missing_' + str(args.n_missing) + '.npy'))
        ssim_pwd=np.load(os.path.join(dataset_path, 'ssim_pwd_missing_' + str(args.n_missing) + '.npy'))
    if not args.pwd_apwd:
        nmse_pwd_apwd=np.load(os.path.join(dataset_path, 'nmse_pwd_apwd_missing_' + str(args.n_missing) + '.npy'))
        ssim_pwd_apwd=np.load(os.path.join(dataset_path, 'ssim_pwd_apwd_missing_' + str(args.n_missing) + '.npy'))
    if not args.wfs:
        nmse_wfs= np.load(os.path.join(dataset_path, 'nmse_wfs_missing_' + str(args.n_missing) + '.npy'))
        ssim_wfs=np.load(os.path.join(dataset_path, 'ssim_wfs_missing_' + str(args.n_missing) + '.npy'))
    if not args.awfs:
        nmse_awfs= np.load(os.path.join(dataset_path, 'nmse_awfs_missing_' + str(args.n_missing) + '.npy'))
        ssim_awfs= np.load(os.path.join(dataset_path, 'ssim_awfs_missing_' + str(args.n_missing) + '.npy'))
    if not args.pm:
        nmse_pwd_pm=np.load(os.path.join(dataset_path, 'nmse_pwd_pm_missing_' + str(args.n_missing) + '.npy'))
        ssim_pwd_pm= np.load(os.path.join(dataset_path, 'ssim_pwd_pm_missing_' + str(args.n_missing) + '.npy'))
    if not args.pwd_cnn:
        nmse_pwd_cnn=np.load(os.path.join(dataset_path, 'nmse_pwd_cnn_missing_' + str(args.n_missing) + '.npy'))
        ssim_pwd_cnn=np.load(os.path.join(dataset_path, 'ssim_pwd_cnn_missing_' + str(args.n_missing) + '.npy'))

    print('save data')
    # Save arrays
    np.savez(os.path.join(dataset_path, 'nmse_missing_' + str(args.n_missing) + '.npz'),
             nmse_pwd=nmse_pwd, nmse_pwd_cnn=nmse_pwd_cnn, nmse_pwd_pm=nmse_pwd_pm, nmse_wfs=nmse_wfs,
             nmse_awfs=nmse_awfs, nmse_pwd_apwd=nmse_pwd_apwd)
    np.savez(os.path.join(dataset_path, 'ssim_missing_' + str(args.n_missing) + '.npz'),
             ssim_pwd=ssim_pwd, ssim_pwd_cnn=ssim_pwd_cnn, ssim_pwd_pm=ssim_pwd_pm, ssim_wfs=ssim_wfs,
             ssim_awfs=ssim_awfs, ssim_pwd_apwd=ssim_pwd_apwd)



if __name__ == '__main__':
    main()
