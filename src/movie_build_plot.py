import tensorflow as tf
import pylab

from utils import *
import movie
import train
import slim


def get_frame_pair_list(data_dir):
    k = movie.FRAME_WINDOW
    lines = [os.path.join(data_dir, str(i) + ".png") + " " + os.path.join(data_dir, str(i + k) + ".png")
             for i in range(len(os.listdir(data_dir)) - k - 1)]  # 1 is for info.txt
    return lines


read_image_pair = tf_to_regular(tf.string)(movie._read_images)


def save_plot(data_dir, model_file, output_file):
    # frame_interval_sec =

    frame_dirs = get_frame_pair_list(data_dir)

    images1_ph = tf.placeholder(tf.float32, [2] + movie.IMAGE_SHAPE)
    images2_ph = tf.placeholder(tf.float32, [2] + movie.IMAGE_SHAPE)
    net_op = movie.build_net(images1_ph, images2_ph, is_training=False)
    correct_prediction = tf.equal(tf.argmax(net_op, 1), tf.constant([1, 0], dtype=tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.get_default_session()
    saver = tf.train.Saver(var_list=tf.get_collection(slim.variables.VARIABLES_TO_RESTORE))

    sess.run(tf.initialize_all_variables())

    tf.train.start_queue_runners()

    train.load_model(saver, sess, model_file)

    f = open(output_file, 'w')

    for i, dirs in enumerate(frame_dirs):
        my_print('\r%i/%i' % (i, len(frame_dirs)))
        image1, image2 = read_image_pair(dirs)
        image1 = image1.reshape([1] + list(image1.shape))
        image2 = image2.reshape([1] + list(image2.shape))
        im12 = np.concatenate((image1, image2))
        im21 = np.concatenate((image2, image1))
        feed = {images1_ph: im12, images2_ph: im21}
        result = sess.run(accuracy, feed)
        f.write(str(i) + ' ' + str(result) + '\n')

    my_print('\r\n')
    f.close()


def show_plot(plot_file, k=10):
    data = pylab.loadtxt(plot_file, delimiter=' ')
    y = np.convolve(data[:, 1], np.ones(k) / k, mode='same')

    pylab.plot(data[:, 0], y)
    pylab.show()


def main():
    """
    Args: data_dir model_file output_file
    """

    args = sys.argv
    if len(args) > 3:
        data_dir = args[1]
        model_file = args[2]
        output_file = args[3]

        sess = tf.Session()
        with sess.as_default():
            save_plot(data_dir, model_file, output_file)
    else:
        plot_file = args[1]
        k = int(args[2])

        show_plot(plot_file, k)


main()
