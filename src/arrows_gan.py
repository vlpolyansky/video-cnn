import sys
import os
import tensorflow as tf

import arrows
import gan


def main():
    """
    Args: data_dir save_dir logs_dir
    """
    args = sys.argv
    data_dir = 'data/arrows/input_col/'
    save_dir = 'save_gan_arrows/'
    logs_dir = 'logs_gan_arrows/'

    bsize = 16

    sess = tf.Session()

    with sess.as_default():
        train_data, _ = arrows.get_input_producers(data_dir, batch_size=bsize, doubled=False)

        def get_data():
            i1, i2, l = train_data()
            return i1, i2

        gan.run_gan(get_data, arrows.gan_generator, arrows.gan_discriminator, bsize, [128, 128, 3],
                    logs_dir=logs_dir, save_dir=save_dir,
                    need_load=os.path.exists(save_dir), init_rate=0.0005)


main()
