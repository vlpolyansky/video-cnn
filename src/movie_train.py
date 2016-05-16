import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import movie
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
        train_data, test_data = movie.get_input_producers(data_dir)
        train.train(movie.build_net, train_data, test_data, logs_dir=logs_dir, save_dir=save_dir, need_load=True,
                init_rate=0.0005, test_only=False)


main()
