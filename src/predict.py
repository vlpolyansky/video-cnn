import tensorflow as tf

from utils import *
import train


# def showarray(a, name='tmp.png', fmt='png'):
#     a = np.uint8(np.clip(a, 0, 1)*255)[0]
#     # if name is None:
#     #     f = StringIO()
#     #     PIL.Image.fromarray(a).save(f, fmt)
#     #     display(Image(data=f.getvalue()))
#     # else:
#     PIL.Image.fromarray(a).save(name, fmt)


def float_to_uint(a):
    return np.uint8(np.clip(a, 0, 1) * 255)


def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5


def predict(build_net_m, image1_raw, save_dir, iter_n=20, step=1, seed=None, image2_raw=None):
    img1 = np.expand_dims(image1_raw, 0)
    image1 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='image1')
    image2 = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='image2')
    net_op = build_net_m(image1, image2)

    # predict_layer = 'conv6/Conv/BiasAdd'
    # t_obj0 = tf.get_default_graph().get_tensor_by_name(predict_layer + ':0')[:, :, :, 0]
    # t_obj1 = tf.get_default_graph().get_tensor_by_name(predict_layer + ':0')[:, :, :, 1]
    # t_score = tf.reduce_mean(t_obj1) - tf.reduce_mean(t_obj0)# - slim.losses.l2_loss(image1 - image2, 5e-2)

    t_score = net_op[0, 1] # - net_op[0, 0]

    t_grad = tf.gradients(t_score, image2)[0]

    init = tf.initialize_all_variables()

    sess = tf.get_default_session()
    saver = tf.train.Saver()

    sess.run(init)
    train.load_model(saver, sess, save_dir)

    if seed is not None:
        np.random.seed(seed)
    img2 = np.expand_dims(image2_raw, 0)

    for i in range(iter_n):
        pred, g, ts = sess.run([net_op, t_grad, t_score], {image1: img1, image2: img2})

        my_print('iter ' + str(i) + '\n')
        my_print('std ' + str(g.std()) + '\n')
        my_print('score ' + str(ts) + '\n')
        my_print('pred ' + str(pred) + '\n')
        my_print('\n')

        g /= g.std() + 1e-20
        img2 += g * step
    my_print('\n')
    return np.concatenate([(img1[0]), (img2[0])], 1), img1[0] - img2[0]
