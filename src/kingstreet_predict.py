import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage
import PIL.Image
import numpy as np

import predict
import kingstreet
from utils import *


def main():
    """
    Args: image_path save_dir iter_n step
    """
    args = sys.argv
    image_path = args[1]
    save_dir = args[2]
    iter_n = int(args[3])
    step = float(args[4])

    img1 = np.float32(PIL.Image.open(image_path))
    # plt.imshow(img1)
    # plt.show()

    sess = tf.Session()

    with sess.as_default():
        result, diff = predict.predict(kingstreet.build_net, img1, save_dir=save_dir, iter_n=iter_n, step=step, seed=123,
                                 image2_raw=img1.copy())

    showarray(result / 255.0)
    showarray(diff / 255.0)


main()
