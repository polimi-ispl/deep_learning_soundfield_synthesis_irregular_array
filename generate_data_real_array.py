import os
import numpy as np
import librosa
import argparse
import pyroomacoustics as pra
from data_lib import params_real
from data_lib import soundfield_generation_real as sg
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Generate data for linear array setup')
    parser.add_argument('--gt_soundfield', type=bool, help='compute ground truth soundfield',
                        default=False)
    parser.add_argument('--n_missing', type=int, help='number missing loudspeakers',
                        default=44)
    parser.add_argument('--dataset_path', type=str, help='number missing loudspeakers',
                        default='/nas/home/lcomanducci/soundfield_synthesis/dataset/real_array')

    args = parser.parse_args()

    # Dataset contained in https://data.research.uts.edu.au/publication/fad2f4b0c03d11ec91ce05dbccc55a63/
    real_dataset_path = '/nas/home/lcomanducci/soundfield_synthesis/dataset/real/Anechoic Room/ZoneE/PlanarMicrophoneArray'
    RIRs_paths = os.listdir(real_dataset_path)

    # READ RIRS
    rir_data_string = 'AnechoicRoom_ZoneE_PlanarMicrophoneArray'
    green_function_sec_sources_path = os.path.join(args.dataset_path, 'green_function_sec_sources_nl_60_real.npy')
    if os.path.exists(green_function_sec_sources_path):
        G = np.load(os.path.join(args.dataset_path, 'green_function_sec_sources_nl_60_real.npy'))
    else:
        print('LOAD REAL GREEN FUNCTIONS')
        G = np.zeros((params_real.N_mic_array, params_real.N_ldspk_array, params_real.N_freqs), dtype=complex)
        for l in tqdm(range(1, params_real.N_ldspk_array + 1)):
            for m in range(1, params_real.N_mic_array + 1):
                audio_orig, fs_orig = librosa.load(
                    os.path.join(real_dataset_path, rir_data_string + '_L' + str(l) + '_M' + str(m) + '.wav'), sr=None)
                G[m - 1, l - 1] =  np.fft.rfft(audio_orig, n=params_real.nfft)[params_real.f_axis_mapping]
        np.save(os.path.join(args.dataset_path, 'green_function_sec_sources_nl_60_real.npy'), G)

    if args.gt_soundfield:
        print('COMPUTE GT SOUNDFIELD')
        P = np.zeros((len(params_real.src_pos_train), params_real.N_mic_array, params_real.N_freqs), dtype=np.complex64)
        for n_s in tqdm(range(len(params_real.src_pos_train))):
            xs = params_real.src_pos_train[n_s]
            xs = np.concatenate([xs, np.array([0])])
            room = pra.AnechoicRoom(fs=params_real.f_s)
            mic_locs_pra = np.array([params_real.x_mic_array, params_real.y_mic_array, np.zeros_like(params_real.x_mic_array)])
            room.add_microphone(mic_locs_pra)
            room.add_source(xs)
            room.compute_rir()
            for n_p in range(params_real.N_mic_array):
                P[n_s, n_p, :] = np.fft.rfft(room.rir[n_p][0], n=params_real.nfft)[params_real.f_axis_mapping]
        np.save(os.path.join(args.dataset_path,'gt_soundfield_train'), P)

    if args.n_missing>0:
        # Load missing loudspeakers configuration:
        idx_missing_path = os.path.join(args.dataset_path,'setup/lspk_config_nl_60_missing_' + str(args.n_missing) + '.npy')
        if os.path.exists(idx_missing_path):
            idx_missing = np.load(idx_missing_path)
        else:
            idx_missing = np.random.choice(np.arange(0, params_real.N_ldspk_array), size=args.n_missing, replace=False)
            np.save(os.path.join(args.dataset_path,'setup/lspk_config_nl_60_missing_' + str(
                args.n_missing) + '.npy'), idx_missing)
            print('SAVED NEW CONFIG')
        N_lspks_config = params_real.N_ldspk_array - args.n_missing

        theta_l = np.delete(params_real.theta_l, idx_missing)
        N, h, theta_n, trunc_mod_exp_idx = sg.model_based_synthesis_circular_array(N_lspks_config, theta_l)


        # Compute input filters
        print('Compute Filters')
        d_array = np.zeros((len(params_real.src_pos_train), N_lspks_config, params_real.N_freqs), dtype=np.complex64)
        for n_s in tqdm(range(len(params_real.src_pos_train))):
            xs = params_real.src_pos_train[n_s]
            # Herglotz density function - point source
            Phi = sg.herglotz_density_point_source(xs, theta_n, trunc_mod_exp_idx, N, A=1)

            # Multiply filters for herglotz density function
            for n_f in range(params_real.N_freqs):
                d_array[n_s, :, n_f] = (1 / N[n_f]) * np.matmul(np.expand_dims(Phi[n_f], axis=0), h[n_f].transpose())

        np.save(os.path.join(args.dataset_path, 'filters_config_nl_' + str(params_real.N_ldspk_array) + '_missing_' + str(args.n_missing) + '.npy'),d_array)


if __name__=='__main__':
    main()