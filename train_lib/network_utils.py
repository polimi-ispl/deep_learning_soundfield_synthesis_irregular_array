import tensorflow as tf

from cvnn import layers as complex_layers
from tensorflow import Tensor


def cart_parametric_relu(z: Tensor, name=None) -> Tensor:
    """
    Applies Leaky Rectified Linear Unit to both the real and imag part of z
    https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu
    http://robotics.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
    :param z: Input tensor.
    :param alpha: Slope of the activation function at x < 0. Default: 0.2
    :param name: A name for the operation (optional).
    :return: Tensor result of the applied activation function
    """
    return tf.cast(tf.complex(tf.keras.layers.PReLU(name=name)(tf.math.real(z)),
                              tf.keras.layers.PReLU(name=name)(tf.math.imag(z))), dtype=z.dtype)


def filter_compensation_model_wideband_skipped_circular(filter_shape, nfft):
    input_layer = complex_layers.complex_input(shape=(filter_shape, nfft, 1),dtype=tf.complex64)
    x = complex_layers.ComplexConv2D(128, 3, 2, padding='valid',dtype=tf.complex64)(input_layer)
    x = cart_parametric_relu(x)
    x = complex_layers.ComplexDropout(0.5)(x)

    x = complex_layers.ComplexConv2D(256, 3, 2, padding='valid',dtype=tf.complex64)(x)
    x = cart_parametric_relu(x)
    x = complex_layers.ComplexDropout(0.5)(x)

    x2 = x  # Skipped connection 1
    x = complex_layers.ComplexConv2D(512, 3, 2, padding='valid',dtype=tf.complex64)(x)
    x = cart_parametric_relu(x)
    x = complex_layers.ComplexDropout(0.5)(x)

    x = complex_layers.ComplexConv2DTranspose(256, 3, 2, padding='valid',dtype=tf.complex64)(x)
    x = cart_parametric_relu(x)
    x = complex_layers.ComplexDropout(0.5)(x)

    x = tf.keras.layers.add([x, x2])
    x = complex_layers.ComplexConv2DTranspose(128, 3, 2, padding='valid',dtype=tf.complex64)(x)
    x = cart_parametric_relu(x)
    x = complex_layers.ComplexDropout(0.5)(x)

    x = complex_layers.ComplexConv2DTranspose(128, (4, 3), 2, padding='valid',dtype=tf.complex64)(x)
    x = cart_parametric_relu(x)
    x = complex_layers.ComplexDropout(0.5)(x)

    # Output
    x = complex_layers.ComplexConv2DTranspose(1, 3, 1, padding='same',activation='linear',dtype=tf.complex64)(x)

    out = x  #
    return tf.keras.models.Model(inputs=input_layer, outputs=out)

    return tf.keras.models.Model(inputs=input_layer, outputs=out)


