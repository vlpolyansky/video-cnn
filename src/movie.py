import tensorflow as tf

from utils import *
import slim


IMAGE_SHAPE = [160, 384, 3]
BATCH_SIZE = 32
FRAME_WINDOW = 3
FRAME_PACK_SIZE = 150


# INPUT

def _create_fname_producers(data_dir):
    k = FRAME_WINDOW
    lines = [os.path.join(data_dir, str(i) + ".png") + " " + os.path.join(data_dir, str(i + k) + ".png")
             for i in range(len(os.listdir(data_dir)) - k - 1)]  # 1 is for info.txt
    lines = np.array(lines)
    lines = lines[:39000]  # remove captions
    mask = np.empty_like(lines, dtype=bool)
    p = 0.8
    for i in range(0, len(mask), FRAME_PACK_SIZE):
        c = randfloat(0, 1) < p
        mask[i: min(i + FRAME_PACK_SIZE, len(mask))] = c
    # shuffle(lines)
    train_lines = list(lines[mask == True])
    test_lines = list(lines[mask == False])
    cnt = len(lines)
    train_cnt = len(train_lines)
    my_print("Total: %i images, train: %i\n" % (cnt, train_cnt))

    return tf.train.string_input_producer(train_lines), tf.train.string_input_producer(test_lines)


def _read_images(paths):
    with tf.name_scope("read_images"):
        path1, path2 = tf.decode_csv(paths, [[""], [""]], field_delim=" ")
        file_content1 = tf.read_file(path1)
        file_content2 = tf.read_file(path2)
        image1 = tf.cast(tf.image.decode_png(file_content1, channels=3, dtype=tf.uint8), tf.float32)
        image2 = tf.cast(tf.image.decode_png(file_content2, channels=3, dtype=tf.uint8), tf.float32)
        image1.set_shape(IMAGE_SHAPE)
        image2.set_shape(IMAGE_SHAPE)
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
            labels12 = tf.constant(1, tf.int64, [batch_size / 2])
    return images12, images21, labels12


def _producer_to_batch(queue, batch_size, doubled, min_after_dequeue=100):
    image1, image2 = _read_images(queue.dequeue())
    images1, images2, labels = _generate_batch(image1, image2, batch_size, doubled,
                                               min_after_dequeue=min_after_dequeue)
    return images1, images2, labels


def _get_data_batches(data_dir, batch_size, doubled):
    train_queue, test_queue = _create_fname_producers(data_dir)
    return _producer_to_batch(train_queue, batch_size, doubled) + \
        _producer_to_batch(test_queue, batch_size, doubled, batch_size)


def get_input_producers(data_dir, batch_size=BATCH_SIZE, doubled=True):
    images1_train, images2_train, labels_train, images1_test, images2_test, labels_test = \
        _get_data_batches(data_dir, batch_size, doubled)

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
    mean1 = tf.tile(mean1, expansion)
    return (images1 - mean1), (images2 - mean1)


def build_net(images1, images2, is_training=True):
    images1, images2 = normalize_images(images1, images2)
    images = tf.concat(3, [images1, images2])

    wd = 0

    with slim.arg_scope([slim.ops.conv2d], stddev=0.1, weight_decay=wd, is_training=is_training):
        net = slim.ops.repeat_op(1, images, slim.ops.conv2d, 16, [3, 3], batch_norm_params={}, scope='conv1')
        net = slim.ops.max_pool(net, [2, 2], scope='pool1')

        net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 32, [3, 3], batch_norm_params={}, scope='conv2')
        net = slim.ops.max_pool(net, [2, 2], scope='pool2')

        net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 64, [3, 3], batch_norm_params={}, scope='conv3')
        net = slim.ops.max_pool(net, [2, 2], scope='pool3')

        net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 128, [3, 3], batch_norm_params={}, scope='conv4')
        net = slim.ops.max_pool(net, [2, 2], scope='pool4')

        net = slim.ops.repeat_op(2, net, slim.ops.conv2d, 256, [3, 3], batch_norm_params={}, scope='conv5')
        net = slim.ops.max_pool(net, [2, 2], scope='pool5')

        net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 2, [3, 3], activation=None, scope='conv6')

        net = tf.reduce_mean(net, reduction_indices=[1, 2], name="reduce")
        net = tf.nn.softmax(net, name="softmax")

    return net
