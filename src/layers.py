import PIL.Image
import tensorflow as tf
from cStringIO import StringIO
import numpy as np

import train
import slim
from utils import *


TILE_SIZE = 128


def showarray(a, name='tmp.png', fmt='png'):
    a = np.uint8(np.clip(a, 0, 1)*255)[0]
    # if name is None:
    #     f = StringIO()
    #     PIL.Image.fromarray(a).save(f, fmt)
    #     display(Image(data=f.getvalue()))
    # else:
    PIL.Image.fromarray(a).save(name, fmt)


def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5


def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = map(tf.placeholder, argtypes)
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap


# Helper function that uses TF to resize an image
def resize(img, size):
    return tf.image.resize_bilinear(img, size)
resize = tffunc(np.float32, np.int32)(resize)


def calc_grad_tiled(img1, img2, t_grad, images1_ph, images2_ph, tile_size=TILE_SIZE):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations.'''
    sz = tile_size
    h, w = img1.shape[1:3]
    sx, sy = np.random.randint(sz, size=2)
    img1_shift = np.roll(np.roll(img1, sx, 2), sy, 1)
    img2_shift = np.roll(np.roll(img2, sx, 2), sy, 1)
    grad1 = np.zeros_like(img1)
    grad2 = np.zeros_like(img2)
    for y in xrange(0, max(h - sz // 2, sz), sz):
        for x in xrange(0, max(w - sz // 2, sz), sz):
            sub1 = img1_shift[:, y : y + sz, x : x + sz]
            sub2 = img2_shift[:, y : y + sz, x : x + sz]
            g1, g2 = tf.get_default_session().run(t_grad, {images1_ph: sub1, images2_ph: sub2})
            grad1[:, y : y + sz, x : x + sz] = g1
            grad2[:, y : y + sz, x : x + sz] = g2
    return np.roll(np.roll(grad1, -sx, 2), -sy, 1), np.roll(np.roll(grad2, -sx, 2), -sy, 1)


def prepare_graph(build_net_m, save_dir):
    images1_ph, images2_ph, labels_ph = train.create_input_placeholders([None, None, 3])
    net_op = build_net_m(images1_ph, images2_ph, trainable=True)

    init = tf.initialize_all_variables()

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess = tf.get_default_session()
    saver = tf.train.Saver(var_list=tf.get_collection(slim.variables.VARIABLES_TO_RESTORE))

    sess.run(init)
    train.load_model(saver, sess, save_dir)

    def maximize_output(layer_name, channel, octave_n=3, octave_scale=1.4, iter_n=20, step=1.0, seed=None):

        t_obj = tf.get_default_graph().get_tensor_by_name(layer_name + ':0')[:, :, :, channel]
        t_score = tf.reduce_mean(t_obj)
        t_grad = tf.gradients(t_score, [images1_ph, images2_ph])

        if seed is not None:
            np.random.seed(seed)
        img1 = np.random.uniform(64, 192, size=(1, TILE_SIZE, TILE_SIZE, 3))
        img2 = img1.copy()

        for octave in range(octave_n):
            my_print("%s %i %i" % (layer_name, channel, octave))
            if octave > 0:
                hw = np.float32(img1.shape[1:3]) * octave_scale
                img1 = resize(img1, np.int32(hw))
                img2 = resize(img2, np.int32(hw))
            for i in range(iter_n):
                g1, g2 = calc_grad_tiled(img1, img2, t_grad, images1_ph, images2_ph)
                div = np.concatenate((g1, g2)).std() + 1e-8
                g1 /= div
                g2 /= div
                img1 += g1 * step
                img2 += g2 * step
                my_print(' .')
            my_print('\r')
            # showarray(visstd(np.concatenate((img1, img2), 2)))
        my_print('\r')
        return visstd(np.concatenate((img1[0], img2[0]), 1))

    return maximize_output
