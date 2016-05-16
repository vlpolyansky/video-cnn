import os, sys
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def randfloat(a, b):
    return rand.random() * (b - a) + a


def my_print(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def showarray(a):
    a = np.asarray(a)
    if len(a.shape) == 4:
        a = a[0]
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1)*255)
    plt.imshow(a)
    plt.show()


def tf_to_regular(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = map(tf.placeholder, argtypes)
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return tf.get_default_session().run(out, dict(zip(placeholders, args)))
            # return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap