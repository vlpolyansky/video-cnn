import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import arrows
import train


def main():
    """
    Args: data_dir save_dir logs_dir
    """
    args = sys.argv
    data_dir = args[1]
    save_dir = args[2]
    logs_dir = args[3]

    sess = tf.Session()

    with sess.as_default():
        train_data, test_data = arrows.get_input_producers(data_dir)
        train.train(arrows.build_net, train_data, test_data, logs_dir=logs_dir, save_dir=save_dir)


main()
