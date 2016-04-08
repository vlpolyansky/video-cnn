import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage

import predict
import arrows
import numpy as np
import PIL.Image

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

    img = np.float32(PIL.Image.open(image_path))
    img1, img2 = np.split(img, 2, 1)
    # showarray(img1 / 255.0)
    # showarray(img2 / 255.0)

    sess = tf.Session()

    with sess.as_default():
        result, diff = predict.predict(arrows.build_net, img1, save_dir=save_dir, iter_n=iter_n, step=step, seed=123,
                                 image2_raw=img1.copy())

    showarray(result / 255.0)
    showarray((diff - diff.mean()) / 255.0 + 0.5)


main()
