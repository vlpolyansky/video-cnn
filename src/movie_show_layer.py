import sys
import tensorflow as tf
import matplotlib.pyplot as plt

import layers
import movie


def main():
    """
    Args: save_path layer channel
    """
    args = sys.argv
    save_dir = args[1]  # ex: conv2/Conv/BiasAdd
    layer = args[2]
    channel = int(args[3])

    sess = tf.Session()

    with sess.as_default():
        maximize_output_multi = layers.prepare_graph(movie.build_net, save_dir)
        result = maximize_output_multi(layer, channel, octave_n=4, iter_n=500, step=1.0, seed=123)

    plt.imshow(result)
    plt.show()


main()
