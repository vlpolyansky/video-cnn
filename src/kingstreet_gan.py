import sys
import os
import tensorflow as tf

import kingstreet
import gan


def main():
    """
    Args: data_dir save_dir logs_dir
    """
    args = sys.argv
    data_dir = 'data/kingstreet/frames/001/'
    save_dir = 'save_gan_ks/'
    logs_dir = 'logs_gan_ks_2/'

    sess = tf.Session()

    with sess.as_default():
        train_data, _ = kingstreet.get_input_producers(data_dir, batch_size=gan.BATCH_SIZE, doubled=False)

        def get_data():
            i1, i2, l = train_data()
            return i1, i2

        gan.run_gan(kingstreet.build_net, get_data, logs_dir=logs_dir, save_dir=save_dir,
                    need_load=os.path.exists(save_dir), init_rate=0.0005)


main()
