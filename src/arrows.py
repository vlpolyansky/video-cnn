import tensorflow as tf

from utils import *
import slim
from gan import deconv2d


IMAGE_SIZE = 128
BATCH_SIZE = 64


# INPUT

def _create_fname_producers(data_dir):
    lines = map((lambda x: os.path.join(data_dir, x)), os.listdir(data_dir))
    # lines = os.listdir(data_dir)
    cnt = len(lines)
    p = 0.9
    train_cnt = int(cnt * p)
    my_print("Total: %i images, train: %i\n" % (cnt, train_cnt))
    train_lines = lines[:train_cnt]
    test_lines = lines[train_cnt:]
    return tf.train.string_input_producer(train_lines), tf.train.string_input_producer(test_lines)


def _read_image(path):
    with tf.name_scope("read_image"):
        file_content = tf.read_file(path)
        image3c = tf.cast(tf.image.decode_png(file_content, channels=3, dtype=tf.uint8), tf.float32)
        image1, image2 = tf.split(1, 2, image3c)
        image1.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
        image2.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
    return image1, image2


def _generate_batch(image1, image2, batch_size, doubled, min_after_dequeue):
    with tf.name_scope("generate_batch"):
        if doubled:
            images1, images2 = tf.train.shuffle_batch([image1, image2], batch_size=batch_size / 2,
                                                      capacity=min_after_dequeue + batch_size * 4,
                                                      min_after_dequeue=min_after_dequeue,
                                                      num_threads=3)

            images12 = tf.concat(0, [images1, images2])
            images21 = tf.concat(0, [images2, images1])
            labels12 = tf.concat(0, [tf.constant(1, tf.int64, [batch_size / 2]),
                                     tf.constant(0, tf.int64, [batch_size / 2])])
        else:
            images12, images21 = tf.train.shuffle_batch([image1, image2], batch_size=batch_size,
                                                      capacity=min_after_dequeue + batch_size * 4,
                                                      min_after_dequeue=min_after_dequeue,
                                                      num_threads=3)
            labels12 = tf.constant(1, tf.int64, [batch_size])
    return images12, images21, labels12


def _producer_to_batch(queue, batch_size, doubled, min_after_dequeue=100):
    image1, image2 = _read_image(queue.dequeue())
    images1, images2, labels = _generate_batch(image1, image2, batch_size, doubled,
                                               min_after_dequeue=min_after_dequeue)
    return images1, images2, labels


def _get_data_batches(data_dir, batch_size, doubled):
    train_queue, test_queue = _create_fname_producers(data_dir)
    return _producer_to_batch(train_queue, batch_size, doubled) + \
        _producer_to_batch(test_queue, batch_size, doubled, batch_size)


def get_input_producers(data_dir, batch_size=BATCH_SIZE, doubled=True):
    images1_train, images2_train, labels_train, images1_test, images2_test, labels_test = \
        _get_data_batches(data_dir, batch_size=batch_size, doubled=doubled)

    def producer(images1, images2, labels):
            def produce():
                sess = tf.get_default_session()
                return sess.run([images1, images2, labels])

            return produce

    return producer(images1_train, images2_train, labels_train), producer(images1_test, images2_test, labels_test)


# NET

def normalize_images(images1, images2):
    mean1 = tf.reduce_mean(images1, [1, 2, 3], True)
    shape = tf.shape(images1)
    expansion = tf.concat(0, [tf.constant([1]), shape[1:4]])
    # expansion = tf.Print(expansion, [expansion], summarize=10)
    mean1 = tf.tile(mean1, expansion)
    # mean1 = tf.tile(mean1, [1, 128, 128, 3])
    return (images1 - mean1), (images2 - mean1)


def build_net(images1, images2, is_training=True):
    images1, images2 = normalize_images(images1, images2)
    images = tf.concat(3, [images1, images2])

    wd = 0

    with slim.arg_scope([slim.ops.conv2d], stddev=0.01, weight_decay=wd, is_training=is_training):
        net = slim.ops.repeat_op(1, images, slim.ops.conv2d, 48, [3, 3], scope='conv1')
        net = slim.ops.max_pool(net, [2, 2], scope='pool1')
        net = tf.nn.lrn(net, name='lrn1')

        net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 64, [3, 3], scope='conv2')
        net = slim.ops.max_pool(net, [2, 2], scope='pool2')
        net = tf.nn.lrn(net, name='lrn2')

        net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 128, [3, 3], scope='conv3')
        net = slim.ops.max_pool(net, [2, 2], scope='pool3')
        net = tf.nn.lrn(net, name='lrn3')

        net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 256, [3, 3], scope='conv4')
        net = slim.ops.max_pool(net, [2, 2], scope='pool4')
        net = tf.nn.lrn(net, name='lrn4')

        net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 2, [3, 3], activation=None, scope='conv5')

        net = tf.reduce_mean(net, reduction_indices=[1, 2], name="reduce")
        net = tf.nn.softmax(net, name="softmax")

    return net


# GAN

def gan_generator(images):
    wd = 0

    net = images

    with tf.variable_scope('generator'):

        with slim.arg_scope([slim.ops.conv2d, deconv2d], stddev=0.1, weight_decay=wd,
                            is_training=True):

            net = conv1 = slim.ops.conv2d(net, 32, [3, 3], batch_norm_params={}, scope='conv1')
            net = pool1 = slim.ops.max_pool(net, [2, 2], scope='pool1')

            net = conv2 = slim.ops.conv2d(net, 64, [3, 3], batch_norm_params={}, scope='conv2')
            net = pool2 = slim.ops.max_pool(net, [2, 2], scope='pool2')

            net = conv3 = slim.ops.conv2d(net, 128, [3, 3], batch_norm_params={}, scope='conv3')
            net = pool3 = slim.ops.max_pool(net, [2, 2], scope='pool3')

            net = conv4 = slim.ops.conv2d(net, 256, [3, 3], batch_norm_params={}, scope='conv4')
            # net = pool4 = slim.ops.max_pool(net, [2, 2], scope='pool4')

            # net = conv5 = slim.ops.conv2d(net, 128, [3, 3], batch_norm_params={}, scope='conv5')
            # net = pool5 = slim.ops.max_pool(net, [2, 2], scope='pool5')

            # print net.get_shape()

            # net = deconv2d(net, [3, 3], conv4.get_shape(), batch_norm_params={}, scope='deconv5')
            # print net.get_shape()

            # net = tf.concat(3, [net, conv4], name='concat4')
            net = deconv2d(net, [3, 3], conv3.get_shape(), batch_norm_params={}, scope='deconv4')
            # print net.get_shape()

            net = tf.concat(3, [net, conv3], name='concat3')
            net = deconv2d(net, [3, 3], conv2.get_shape(), batch_norm_params={}, scope='deconv3')
            # print net.get_shape()

            net = tf.concat(3, [net, conv2], name='concat2')
            net = deconv2d(net, [3, 3], images.get_shape(), scope='deconv2')
            # print net.get_shape()

            # net = tf.concat(3, [net, pool1], name='concat1')
            # net = deconv2d(net, [3, 3], images.get_shape(), activation=None, scope='deconv1')
            # print net.get_shape()

    return net


def gan_discriminator(images1, images2, reuse=False):
    wd = 0

    images = tf.concat(3, [images1, images2])
    net = images

    with tf.variable_scope('discriminator'):

        with slim.arg_scope([slim.ops.conv2d], stddev=0.1, weight_decay=wd, is_training=True):

            if reuse:
                tf.get_variable_scope().reuse_variables()

            net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 32, [3, 3], batch_norm_params={}, scope='conv1')
            net = slim.ops.max_pool(net, [2, 2], scope='pool1')

            net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 64, [3, 3], batch_norm_params={}, scope='conv2')
            net = slim.ops.max_pool(net, [2, 2], scope='pool2')

            net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 128, [3, 3], batch_norm_params={}, scope='conv3')
            net = slim.ops.max_pool(net, [2, 2], scope='pool3')

            net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 256, [3, 3], batch_norm_params={}, scope='conv4')
            net = slim.ops.max_pool(net, [2, 2], scope='pool4')

            net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 1, [3, 3], activation=None, scope='conv5')

            net = tf.reduce_mean(net, reduction_indices=[1, 2, 3], name='reduce')
            # net = tf.nn.sigmoid(net)

    return net
