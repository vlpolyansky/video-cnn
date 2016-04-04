import tensorflow as tf

from utils import *
import slim


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


def _generate_batch(image1, image2, batch_size, min_after_dequeue):
    with tf.name_scope("generate_batch"):
        images1, images2 = tf.train.shuffle_batch([image1, image2], batch_size=batch_size / 2,
                                                  capacity=min_after_dequeue + batch_size * 4,
                                                  min_after_dequeue=min_after_dequeue,
                                                  num_threads=1)

        images12 = tf.concat(0, [images1, images2])
        images21 = tf.concat(0, [images2, images1])
        labels12 = tf.concat(0, [tf.constant(1, tf.int64, [batch_size / 2]),
                                 tf.constant(0, tf.int64, [batch_size / 2])])
    return images12, images21, labels12


def _producer_to_batch(queue, min_after_dequeue=500):
    image1, image2 = _read_image(queue.dequeue())
    images1, images2, labels = _generate_batch(image1, image2, BATCH_SIZE,
                                               min_after_dequeue=min_after_dequeue)
    return images1, images2, labels


def _get_data_batches(data_dir):
    train_queue, test_queue = _create_fname_producers(data_dir)
    return _producer_to_batch(train_queue) + \
        _producer_to_batch(test_queue, BATCH_SIZE)


def get_input_producers(data_dir):
    images1_train, images2_train, labels_train, images1_test, images2_test, labels_test = \
        _get_data_batches(data_dir)

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


def build_net(images1, images2, trainable=True):
    images1, images2 = normalize_images(images1, images2)
    images = tf.concat(3, [images1, images2])

    wd = 0

    with slim.arg_scope([slim.ops.conv2d], stddev=0.01, weight_decay=wd, trainable=trainable):
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
