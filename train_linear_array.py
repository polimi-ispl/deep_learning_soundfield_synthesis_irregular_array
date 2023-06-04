import numpy as np
import os
import argparse
import datetime
def main():

    parser = argparse.ArgumentParser(
        description='Soundfield reconstruction')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=5000)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--log_dir', type=str, help='Tensorboard log directory',
                        default='/nas/home/lcomanducci/soundfield_synthesis_RQ/logs/scalars')
    parser.add_argument('--n_missing', type=int, help='number of missing loudspeakers',default=16)
    parser.add_argument('--gt_soundfield_dataset_path', type=str, help='path to dataset',
                        default='/nas/home/lcomanducci/soundfield_synthesis/dataset/linear_array/gt_soundfield_train' )
    parser.add_argument('--learning_rate', type=float, help='LEarning rate', default=0.0001)
    parser.add_argument('--green_function',
        type=str,
        help='LEarning rate',
        default='/nas/home/lcomanducci/soundfield_synthesis/dataset/linear_array/green_function_sec_sources_nl_64.npy')
    parser.add_argument('--gpu', type=str, help='GPU number', default='0')

    args = parser.parse_args()
    number_missing_loudspeakers = args.n_missing
    epochs = args.epochs
    batch_size = args.batch_size
    log_dir = args.log_dir
    gt_soundfield_dataset_path = args.gt_soundfield_dataset_path
    lr = args.learning_rate

    os.environ['CUDA_ALLOW_GROWTH'] = 'True'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import scipy
    import sfs
    from train_lib import network_utils
    from train_lib import train_utils
    from data_lib import params_linear
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    early_stop_patience = 10
    # Construct paths
    filter_dataset_path = '/nas/home/lcomanducci/soundfield_synthesis/dataset/linear_array/filters_config_nl_64_missing_'\
                          +str(number_missing_loudspeakers)+'_cp'+str(len(params_linear.idx_cp))+'.npy'
    mask_path = '/nas/home/lcomanducci/soundfield_synthesis/dataset/linear_array/setup/lspk_config_nl_64_missing_'\
                +str(number_missing_loudspeakers)+'.npy'
    saved_model_path = \
        '/nas/home/lcomanducci/soundfield_synthesis_RQ/models/linear_array/model_linear_config_nl_64_missing_'\
        +str(number_missing_loudspeakers)+'_COMPLEX_CP_'+str(len(params_linear.point_cp))+'_lr_'+str(lr)+'PReLU_earlyStop_'+str(early_stop_patience)
    log_name = 'linear_array_config_nl_64_missing_'+str(number_missing_loudspeakers)+ '_lr_' + str(lr)+'PReLU_earlyStop_'+str(early_stop_patience)

    # Tensorboard and logging
    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + log_name)
    summary_writer = tf.summary.create_file_writer(log_dir)


    # Training params
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)#,clipnorm=100)

    epoch_to_plot = 25  # Plot evey epoch_to_plot epochs
    val_perc = 0.2

    N_mics = 64
    filter_shape = int(N_mics * 2)
    nfft = int(129)

    # Load configuration
    idx_missing = np.load(mask_path)
    # Load Green function
    G = np.load(args.green_function).astype(np.complex64)
    d_ = np.load(filter_dataset_path).astype(np.complex64)  # Filters

    import time
    tic = time.time()
    gt_soundfield_dataset_path = \
        '/nas/home/lcomanducci/soundfield_synthesis/dataset/linear_array/gt_soundfield_train'+'_cp' \
        + str(len(params_linear.idx_cp)) + '.npy'

    P_gt = np.load(gt_soundfield_dataset_path).astype(np.complex64)   # gt soundfield
    print(str(time.time()-tic) + ' seconds lost for lazy data loading')

    G_cp = G[params_linear.idx_lr[params_linear.idx_cp]]
    G = np.delete(G, idx_missing, axis=1)
    G_cp = np.delete(G_cp, idx_missing, axis=1)
    G = tf.convert_to_tensor(G)
    G_cp = tf.convert_to_tensor(G_cp)
    # Load dataset


    # Split train/val
    d_train, d_val, P_train, P_val, src_train, src_val = \
        train_test_split(d_, P_gt, params_linear.src_pos_train.transpose(), test_size=val_perc)

    do_overfit = False
    if do_overfit:
        epoch_to_plot = 100
        d_train = d_train[:100]
        P_train = P_train[:100]
        src_train = src_train[:100]

        d_val = d_train
        P_val = P_train
        src_val = src_train

    def preprocess_dataset(d, P, src):
        data_ds = tf.data.Dataset.from_tensor_slices((d,P, src))
        return data_ds

    train_ds = preprocess_dataset(d_train, P_train, src_train)
    val_ds = preprocess_dataset(d_val, P_val, src_val)

    #loss_fn = ComplexMAE()
    loss_fn = tf.keras.losses.MeanAbsoluteError()
    filter_shape = int(d_train.shape[1])
    N_freqs = params_linear.N_freqs

    # Load Network
    network_model_filters = network_utils.filter_compensation_model_wideband_skipped_circular(filter_shape, N_freqs)
    #network_model_filters = tf.keras.models.load_model(saved_model_path)
    network_model_filters.summary()

    # Load Data
    train_ds = train_ds.shuffle(buffer_size=int(batch_size*2))
    val_ds = val_ds.shuffle(buffer_size=int(batch_size*2))

    train_ds = train_ds.batch(batch_size=batch_size)
    val_ds = val_ds.batch(batch_size=batch_size)

    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    class NMSEMetric(tf.keras.metrics.Metric):
        def __init__(self, name="categorical_true_positives", **kwargs):
            super(NMSEMetric, self).__init__(name=name, **kwargs)
            self.nmse = self.add_weight(name="ctp", initializer="zeros")
        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.reshape(y_true, (-1, len(params_linear.idx_cp), params_linear.N_freqs))
            y_pred = tf.reshape(y_pred, (-1, len(params_linear.idx_cp), params_linear.N_freqs))
            mse = tf.math.pow(tf.math.abs(y_true-y_pred),2)
            norm = tf.math.pow(tf.math.abs(y_true),2)
            nmse = tf.divide(tf.reduce_sum(mse,axis=1),tf.reduce_sum(norm,axis=1))
            if nmse is not None:
                nmse = tf.cast(nmse, "float32")
            self.nmse.assign_add(tf.reduce_mean(nmse))
        def result(self):
            return self.nmse
        def reset_state(self):
            # The state of the metric will be reset at the start of each epoch.
            self.nmse.assign(0.0)

    metric_fn_train = NMSEMetric()
    metric_fn_val = NMSEMetric()

    @tf.function
    def train_step(d_, P_):
        with tf.GradientTape() as tape:
            # Compensate driving signals
            d_hat = network_model_filters(d_, training=True)[:, :, :, 0]
            p_est = tf.einsum('bij, kij-> bkj', d_hat, G_cp)
            # Reshape for easier loss computation
            P_= tf.reshape(P_, (-1, len(params_linear.idx_cp) * params_linear.N_freqs))
            p_est = tf.reshape(p_est, (-1, len(params_linear.idx_cp) * params_linear.N_freqs))

            loss_value_P = loss_fn(P_,p_est)

        network_model_filters_grads = tape.gradient(loss_value_P, network_model_filters.trainable_weights)
        optimizer.apply_gradients(zip(network_model_filters_grads, network_model_filters.trainable_weights))


        metric_fn_train.update_state(P_,p_est)
        return loss_value_P

    @tf.function
    def val_step(d_, P_):
            # Compensate driving signals
            d_hat = network_model_filters(d_, training=False)[:, :, :, 0]
            p_est = tf.einsum('bij, kij-> bkj', d_hat, G_cp)
            loss_value_P = loss_fn(tf.reshape(P_,(-1,len(params_linear.idx_cp)*params_linear.N_freqs)),
                                   tf.reshape(p_est,(-1,len(params_linear.idx_cp)*params_linear.N_freqs)))
            metric_fn_val.update_state(P_, p_est)
            return loss_value_P, d_hat

    for n_e in tqdm(range(epochs)):
        plot_val = True

        n_step = 0
        train_loss = 0
        for d, P, _ in train_ds:
            train_loss = train_loss + train_step(d, P)
            n_step = n_step + 1
        train_loss = train_loss/n_step


        n_step = 0
        val_loss = 0
        for d, P, src in val_ds:
            loss_value_P_val, d_hat = val_step(d, P)
            val_loss = val_loss + loss_value_P_val
            n_step = n_step + 1
        val_loss = val_loss / n_step

        # Log to tensorboard
        with summary_writer.as_default():
            tf.summary.scalar('train_loss_P', train_loss, step=n_e)
            tf.summary.scalar('val_loss_P', val_loss, step=n_e)

            tf.summary.scalar('train_nmse_metric',10*tf.experimental.numpy.log10(metric_fn_train.result()), step=n_e)
            tf.summary.scalar('val_nmse_metric',10*tf.experimental.numpy.log10(metric_fn_val.result()), step=n_e)
        metric_fn_train.reset_states()
        metric_fn_val.reset_states()

        # Every epoch_to_plot epochs plot an example of validation
        if not n_e % epoch_to_plot and plot_val:
            print('Train loss: ' + str(train_loss.numpy()))
            print('Val loss: ' + str(val_loss.numpy()))

            n_s = np.random.randint(0, src.shape[0])
            idx_f = 41
            P_hat_real =  tf.math.real(tf.einsum('bij, kij-> bkj', d_hat, G))[n_s]
            P_pwd_real =  tf.math.real(tf.einsum('bij, kij-> bkj', d, G))[n_s]
            P_gt = np.zeros(P_hat_real.shape, dtype=complex)
            for n_f in range(N_freqs):
                P_gt[ :, n_f] = (1j / 4) * \
                               scipy.special.hankel2(0,
                                                     (params_linear.wc[n_f] / params_linear.c) *
                                                     np.linalg.norm(params_linear.point[:, :2] - src[n_s], axis=1))

            p_pwd_real = np.reshape(P_pwd_real[:, idx_f], (params_linear.N_sample, params_linear.N_sample))
            p_hat_real = np.reshape(P_hat_real[:, idx_f], (params_linear.N_sample, params_linear.N_sample))
            p_gt = np.reshape(np.real(P_gt[ :, idx_f]), (params_linear.N_sample, params_linear.N_sample))
            selection = np.ones_like(params_linear.array_pos[:, 0])
            selection[idx_missing] = 0
            figure_soundfield = plt.figure(figsize=(10, 20))
            plt.subplot(311)
            plt.imshow(p_pwd_real,aspect='auto',cmap='magma'),plt.colorbar()
            plt.title('pwd + F:' + str(params_linear.f_axis[idx_f]) + ' Hz')
            plt.subplot(312)
            plt.imshow(p_gt, aspect='auto', cmap='magma'),plt.colorbar()
            plt.title('GT')
            plt.subplot(313)
            plt.imshow(p_hat_real,aspect='auto',cmap='magma'),plt.colorbar()
            plt.title('est')
            plt.show()

            with summary_writer.as_default():
                tf.summary.image("soundfield second training", train_utils.plot_to_image(figure_soundfield), step=n_e)

            filters_fig = plt.figure()
            plt.plot(d_hat.numpy()[0, :, :])
            with summary_writer.as_default():
                tf.summary.image("Filters true second", train_utils.plot_to_image(filters_fig), step=n_e)

        # Select best model
        if n_e == 0:
            lowest_val_loss = val_loss
            network_model_filters.save(saved_model_path)
            early_stop_counter = 0

        else:
            if val_loss < lowest_val_loss:
                network_model_filters.save(saved_model_path)
                print('Model updated')
                lowest_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter = early_stop_counter + 1
                print('Patience status: '+str(early_stop_counter)+'/'+str(early_stop_patience))

        # Early stopping
        if early_stop_counter > early_stop_patience:
            print('Training finished at epoch '+str(n_e))
            break



if __name__ == '__main__':
    main()

