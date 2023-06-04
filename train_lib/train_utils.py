import io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def normalize(x):
    x_mean = np.mean(x)
    x_std_dev = np.sqrt(np.var(x))
    x_norm = (x - x_mean) / x_std_dev
    x_norm = x
    return x_norm, x_mean, x_std_dev


def normalize_graph(x):
    x_mean = tf.math.reduce_mean(x)
    x_std_dev = tf.math.sqrt(tf.math.reduce_variance(x))
    x_norm = (x - x_mean) / x_std_dev
    return x_norm #, x_mean, x_std_dev


def denormalize(x_norm, x_mean, x_std_dev):
    x_denorm = (x_norm * x_std_dev) + x_mean
    x_denorm = x_norm
    return x_denorm

def normalize_tensor(input_tensor, do_normalize=False):
    """ Normalizes 3D tensor between 0 and 1
    Args:
        input_tensor, tf.Tensor
    Returns:
        norm_tensor, tf.Tensor
    """
    max_val = tf.math.reduce_max(tf.reshape(input_tensor, shape=[input_tensor.shape[0], -1])) #), axis=-1)
    min_val = tf.math.reduce_min(tf.reshape(input_tensor, shape=[input_tensor.shape[0], -1])) #, axis=-1)
    # norm_tensor = tf.divide(input_tensor - tf.expand_dims(tf.expand_dims(tf.expand_dims(min_val, 1), 2),3),
    #                        tf.expand_dims(tf.expand_dims(tf.expand_dims(max_val - min_val, 1), 2),3))
    if do_normalize:
        return tf.divide(input_tensor - min_val, max_val - min_val)
    else:
        return input_tensor

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image




