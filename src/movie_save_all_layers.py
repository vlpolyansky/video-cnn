import sys
import tensorflow as tf
import matplotlib.pyplot as plt

import layers
import movie
from utils import *


def main():
    """
    Args: save_path output_dir
    """
    args = sys.argv
    save_dir = args[1]
    output_dir = args[2]

    layer_list = [
        'conv1/Conv/Conv2D',
        'conv2/Conv/Conv2D',
        'conv3/Conv/Conv2D',
        'conv4/Conv/Conv2D',
        'conv5/Conv/Conv2D',
        'conv5/Conv_1/Conv2D',
        'conv6/Conv/Conv2D'
    ]
    channels = [16, 32, 64, 64, 128, 256, 2]

    sess = tf.Session()

    with sess.as_default():
        maximize_output_multi = layers.prepare_graph(movie.build_net, save_dir)

        for i, layer in enumerate(layer_list):
            folder_name = layer.replace('/', '_')
            directory = os.path.join(output_dir, folder_name)
            create_dir(directory)
            for channel in range(channels[i]):
                result = maximize_output_multi(layer, channel, octave_n=4, iter_n=100, step=5.0, seed=123)
                plt.imsave(os.path.join(directory, str(channel) + '.png'), result)


main()
