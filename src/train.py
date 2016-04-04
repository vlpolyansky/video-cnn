import tensorflow as tf

from utils import *
import slim


def create_input_placeholders(image_size):
    images1_ph = tf.placeholder(tf.float32, [None] + image_size, "images1")
    images2_ph = tf.placeholder(tf.float32, [None] + image_size, "images2")
    labels_ph = tf.placeholder(tf.int64, [None], "labels")
    return images1_ph, images2_ph, labels_ph


def dense_to_one_hot(label_batch, num_labels=2):
    with tf.name_scope("one_hot_encoder"):
        sparse_labels = tf.cast(tf.reshape(label_batch, [-1, 1]), tf.int32)
        derived_size = tf.shape(sparse_labels)[0]
        indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
        concated = tf.concat(1, [indices, sparse_labels])
        outshape = tf.concat(0, [tf.reshape(derived_size, [1]), tf.reshape(num_labels, [1])])
        labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
        labels = tf.cast(labels, tf.float32)
    return labels


def build_loss(logits, labels):
    with tf.name_scope("cross_entropy"):
        labels = tf.cast(labels, tf.int64)
        dense_labels = dense_to_one_hot(labels)
        clipped_logits = tf.clip_by_value(logits, 1e-10, 100.0)
        cross_entropy = -dense_labels * tf.log(clipped_logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection(slim.losses.LOSSES_COLLECTION, cross_entropy_mean)

    return cross_entropy_mean


def get_total_loss(losses_list):
    total_loss = tf.add_n(losses_list, name='total_loss')
    loss_summary = calc_loss_summaries(losses_list + [total_loss])
    return total_loss, loss_summary


def calc_loss_summaries(losses_list):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses_list)
    for l in losses_list:
        tf.scalar_summary(l.op.name + '/raw', l)
        tf.scalar_summary(l.op.name + '/avg', loss_averages.average(l))

    return loss_averages_op


def calc_accuracy(prediction, labels_sparse):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), labels_sparse)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_summary = tf.scalar_summary("accuracy", accuracy)

    return accuracy, accuracy_summary


def build_optimizer(loss_op, step, init_rate):

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(init_rate)
        grads = optimizer.compute_gradients(loss_op)

        apply_gradient_op = optimizer.apply_gradients(grads, global_step=step)

        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

    return apply_gradient_op


def load_model(saver, sess, path):
    saver.restore(sess, path)
    my_print("Model restored.\n")


def save_model(saver, sess, path):
    save_path = saver.save(sess, path)
    my_print("Model saved in file: %s\n" % save_path)


def train(build_net_m, get_train_data_m, get_test_data_m, iter_n=10000000, init_rate=0.0001, logs_dir='logs_tmp/',
          save_dir='save_tmp/', need_load=False):
    create_dir(logs_dir)
    create_dir(save_dir)
    save_file_path = os.path.join(save_dir, 'model.ckpt')

    step = slim.variables.variable('step', [], tf.int32, tf.constant_initializer(0), trainable=False)

    images1_ph, images2_ph, labels_ph = create_input_placeholders([None, None, 3])
    net_op = build_net_m(images1_ph, images2_ph, trainable=True)

    _ = build_loss(net_op, labels_ph)
    total_loss, loss_summary = get_total_loss(tf.get_collection(slim.losses.LOSSES_COLLECTION))

    with tf.control_dependencies([loss_summary]):
        train_op = build_optimizer(total_loss, step, init_rate=init_rate)

    accuracy, accuracy_summary = calc_accuracy(net_op, labels_ph)

    merged_summaries = tf.merge_all_summaries()

    sess = tf.get_default_session()
    saver = tf.train.Saver(var_list=tf.get_collection(slim.variables.VARIABLES_TO_RESTORE))
    writer = tf.train.SummaryWriter(logs_dir, sess.graph, flush_secs=30)

    sess.run(tf.initialize_all_variables())

    tf.train.start_queue_runners()

    if need_load:
        load_model(saver, sess, save_file_path)

    my_print("Starting...\n")

    for i in range(0, iter_n):
        if i % 11 == 0:
            # Test
            im1, im2, lab = get_test_data_m()
            feed = {
                images1_ph: im1,
                images2_ph: im2,
                labels_ph: lab
            }
            result = sess.run([accuracy, step], feed)
            acc = result[0]
            st = result[1]
            print("\rAccuracy on test on step %s: %s" % (st, acc))
        else:
            # Train
            im1, im2, lab = get_train_data_m()
            feed = {
                images1_ph: im1,
                images2_ph: im2,
                labels_ph: lab
            }
            result = sess.run([train_op, merged_summaries, step], feed)
            summary_str = result[1]
            st = result[2]
            if st % 10 == 0:
                writer.add_summary(summary_str, st)

            if i % 100 == 0:
                save_model(saver, sess, save_file_path)
