import tensorflow as tf
from data_lib import params_linear

def filter_compensation_model_wideband_skipped_circular(filter_shape, nfft):

    input_layer = tf.keras.layers.Input(shape=(filter_shape, nfft, 1))
    x = tf.keras.layers.Conv2D(128, 3, 2, padding='valid')(input_layer)
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(256, 3, 2, padding='valid')(x)
    x = tf.keras.layers.PReLU()(x)
    x2 = x  # Skipped connection 1
    x = tf.keras.layers.Conv2D(512, 3, 2, padding='valid')(x)
    x = tf.keras.layers.PReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(256, 3, 2, padding='valid')(x)
    x = tf.keras.layers.PReLU()(x)

    x = tf.keras.layers.add([x, x2])
    x = tf.keras.layers.Conv2DTranspose(128, 3, 2, padding='valid')(x)
    x = tf.keras.layers.PReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(128, (4, 3), 2, padding='valid')(x)
    x = tf.keras.layers.PReLU()(x)

    # Output
    x = tf.keras.layers.Conv2DTranspose(1, 3, 1, padding='same')(x)
    out = x  #
    return tf.keras.models.Model(inputs=input_layer, outputs=out)