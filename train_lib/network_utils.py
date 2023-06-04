import tensorflow as tf
from cvnn import layers as complex_layers
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

"""
def filter_compensation_model_wideband_skipped_circular_COMPLEX(filter_shape, nfft):

    input_layer = complex_layers.complex_input(shape=(filter_shape, nfft, 1))
    x = complex_layers.ComplexConv2D(128, 3, 2, padding='valid',activation='cart_relu')(input_layer)
    #x = complex_layers.ComplexConv2D(128, 3, 1, padding='same',activation='cart_leaky_relu')(x)

    x = complex_layers.ComplexConv2D(256, 3, 2, padding='valid',activation='cart_relu')(x)
    #x = complex_layers.ComplexConv2D(256, 3, 1, padding='same',activation='cart_leaky_relu')(x)


    x = complex_layers.ComplexConv2D(512, 3, 2, padding='valid',activation='cart_relu')(x)
    #x = complex_layers.ComplexConv2D(512, 3, 1, padding='same',activation='cart_leaky_relu')(x)


    x = complex_layers.ComplexConv2DTranspose(512, 3, 2, padding='valid',activation='cart_relu')(x)
    #x = complex_layers.ComplexConv2DTranspose(512, 3, 1, padding='same',activation='cart_leaky_relu')(x)

    x = complex_layers.ComplexConv2DTranspose(256, 3, 2, padding='valid',activation='cart_relu')(x)
    #x = complex_layers.ComplexConv2DTranspose(256, 3, 1, padding='same',activation='cart_leaky_relu')(x)


    x = complex_layers.ComplexConv2DTranspose(128, (4, 3), 2, padding='valid',activation='cart_relu')(x)
    #x = complex_layers.ComplexConv2DTranspose(128, 3, 1, padding='same',activation='cart_leaky_relu')(x)


    out = complex_layers.ComplexConv2DTranspose(1, 3, 1, padding='same',activation='linear')(x)
    return tf.keras.models.Model(inputs=input_layer, outputs=out)
"""
"""
def filter_compensation_model_wideband_skipped_circular_COMPLEX(filter_shape, nfft):

    input_layer = complex_layers.complex_input(shape=(filter_shape, nfft, 1))
    x = complex_layers.ComplexConv2D(128, 3, 2, padding='valid',activation='cart_leaky_relu')(input_layer)
    x = complex_layers.ComplexConv2D(256, 3, 2, padding='valid',activation='cart_leaky_relu')(x)
    x2 = x  # Skipped connection 1
    x = complex_layers.ComplexConv2D(512, 3, 2, padding='valid',activation='cart_leaky_relu')(x)

    x = complex_layers.ComplexConv2DTranspose(256, 3, 2, padding='valid',activation='cart_leaky_relu')(x)

    x = tf.keras.layers.add([x, x2])
    x = complex_layers.ComplexConv2DTranspose(128, 3, 2, padding='valid',activation='cart_leaky_relu')(x)

    x = complex_layers.ComplexConv2DTranspose(128, (4, 3), 2, padding='valid',activation='cart_leaky_relu')(x)

    # Output
    x = complex_layers.ComplexConv2DTranspose(1, 3, 1, padding='same',activation='linear')(x)
    out = x  #
    return tf.keras.models.Model(inputs=input_layer, outputs=out)

"""
def filter_compensation_model_wideband_skipped_circular_COMPLEX(filter_shape, nfft):

    input_layer = complex_layers.complex_input(shape=(filter_shape, nfft, 1))
    x = complex_layers.ComplexConv2D(128, 3, 2, padding='valid',activation='cart_leaky_relu')(input_layer)
    #x = complex_layers.ComplexConv2D(128, 3, 1, padding='same',activation='cart_relu')(x)

    x = complex_layers.ComplexConv2D(256, 3, 2, padding='valid',activation='cart_leaky_relu')(x)
    #x = complex_layers.ComplexConv2D(256, 3, 1, padding='same', activation='cart_relu')(x)
    x2 = x  # Skipped connection 1
    x = complex_layers.ComplexConv2D(512, 3, 2, padding='valid',activation='cart_leaky_relu')(x)
    #x = complex_layers.ComplexConv2D(512, 3, 1, padding='same', activation='cart_relu')(x)

    x = complex_layers.ComplexConv2DTranspose(256, 3, 2, padding='valid',activation='cart_leaky_relu')(x)
    #x = complex_layers.ComplexConv2D(256, 3, 1, padding='same', activation='cart_relu')(x)

    x = tf.keras.layers.add([x, x2])
    x = complex_layers.ComplexConv2DTranspose(128, 3, 2, padding='valid',activation='cart_leaky_relu')(x)
    #x = complex_layers.ComplexConv2D(128, 3, 1, padding='same', activation='cart_relu')(x)

    x = complex_layers.ComplexConv2DTranspose(128, (4, 3), 2, padding='valid',activation='cart_leaky_relu')(x)
    #x = complex_layers.ComplexConv2D(128, 3, 1, padding='same', activation='cart_relu')(x)

    # Output
    x = complex_layers.ComplexConv2DTranspose(1, 3, 1, padding='same')(x)
    out = x  #
    return tf.keras.models.Model(inputs=input_layer, outputs=out)
"""
def filter_compensation_model_wideband_skipped_circular_COMPLEX_REAL_DATA(filter_shape, nfft):

    input_layer = complex_layers.complex_input(shape=(filter_shape, nfft, 1))
    x = complex_layers.ComplexConv2D(128, 3, 2, padding='valid',activation='cart_relu')(input_layer)
    x = complex_layers.ComplexConv2D(128, 3, 1, padding='same',activation='cart_relu')(x)

    x = complex_layers.ComplexConv2D(256, 3, 2, padding='valid',activation='cart_relu')(x)
    x = complex_layers.ComplexConv2D(256, 3, 1, padding='same', activation='cart_relu')(x)
    x2 = x  # Skipped connection 1
    x = complex_layers.ComplexConv2D(512, 3, 2, padding='valid',activation='cart_relu')(x)
    x = complex_layers.ComplexConv2D(512, 3, 1, padding='same', activation='cart_relu')(x)

    x = complex_layers.ComplexConv2DTranspose(256, 3, 2, padding='valid',activation='cart_relu')(x)
    x = complex_layers.ComplexConv2D(256, 3, 1, padding='same', activation='cart_relu')(x)

    x = tf.keras.layers.add([x, x2])
    x = complex_layers.ComplexConv2DTranspose(128, 3, 2, padding='valid',activation='cart_relu')(x)
    x = complex_layers.ComplexConv2D(128, 3, 1, padding='same', activation='cart_relu')(x)

    x = complex_layers.ComplexConv2DTranspose(128, (4, 3), 2, padding='valid',activation='cart_relu')(x)
    x = complex_layers.ComplexConv2D(128, 3, 1, padding='same', activation='cart_relu')(x)

    # Output
    x = complex_layers.ComplexConv2DTranspose(1, 3, 1, padding='same',activation='linear')(x)
    out = x  #
    return tf.keras.models.Model(inputs=input_layer, outputs=out)

def PRESSURE_MATCHING_COMPLEX(sf_shape, nfft, filter_shape):

    # Encoder
    input_layer = complex_layers.complex_input(shape=(sf_shape, nfft, 1))
    x = complex_layers.ComplexConv2D(64, 3, 2, padding='same')(input_layer)
    x = complex_layers.ComplexConv2D(64, 3, 2, padding='same',activation='cart_relu')(x)
    x = complex_layers.ComplexConv2D(128, 3, 2, padding='same',activation='cart_relu')(x)
    x = complex_layers.ComplexConv2D(128, 3, 2, padding='same',activation='cart_relu')(x)
    x = complex_layers.ComplexConv2D(256, 3, 2, padding='same',activation='cart_relu')(x)
    x = complex_layers.ComplexConv2D(256, 3, 2, padding='same',activation='cart_relu')(x)
    x = complex_layers.ComplexFlatten()(x)

    # Bottleneck
    depth_decoder = 4
    bottleneck_dim_filters = 4
    bottleneck_dim_freq = 8
    x = complex_layers.ComplexDense(bottleneck_dim_filters*bottleneck_dim_freq)(x)
    x = tf.keras.layers.Reshape((bottleneck_dim_filters, bottleneck_dim_freq,1))(x)

    # Decoder
    x = complex_layers.ComplexConv2DTranspose(256, 3, 2, padding='same',activation='cart_relu')(x)
    x = complex_layers.ComplexConv2DTranspose(256, 3, 1, padding='same',activation='cart_relu')(x)
    x = complex_layers.ComplexConv2DTranspose(128, 3, 2, padding='same',activation='cart_relu')(x)
    x = complex_layers.ComplexConv2DTranspose(128, 3, 1, padding='same',activation='cart_relu')(x)
    x = complex_layers.ComplexConv2DTranspose(64, 3, 2, padding='same',activation='cart_relu')(x)
    x = complex_layers.ComplexConv2DTranspose(64, 3, 1, padding='same',activation='cart_relu')(x)

    # Output
    x = tf.keras.layers.Cropping2D(((0,0),(0,1)))(x)
    out = complex_layers.ComplexConv2DTranspose(1, 3, 1, padding='same',activation='linear')(x)

    return tf.keras.models.Model(inputs=input_layer, outputs=out)
"""