import tensorflow as tf

import slim
import train
from utils import *


# IMAGE_SHAPE = [432, 768, 3]
# IMAGE_SHAPE = [128, 128, 3]
# BATCH_SIZE = 4


@slim.scopes.add_arg_scope
def deconv2d(inputs,
             kernel_size,
             output_shape,
             stride=2,
             padding='SAME',
             activation=tf.nn.relu,
             stddev=0.1,
             bias=0.0,
             weight_decay=0,
             batch_norm_params=None,
             is_training=True,
             trainable=True,
             restore=True,
             scope=None):
    if len(kernel_size) != 2:
        raise ValueError('kernel_size must be a 2-D list.')
    with tf.variable_op_scope([inputs], scope, 'Deconv'):
        num_filters_in = inputs.get_shape()[-1]
        weights_shape = [kernel_size[0], kernel_size[1],
                         output_shape[-1], num_filters_in]
        weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
        # l2_regularizer = lambda t: losses.l2_loss(t, weight_decay)
        weights = slim.variables.variable('weights',
                                     shape=weights_shape,
                                     initializer=weights_initializer,
                                     trainable=trainable)
        deconv = tf.nn.conv2d_transpose(inputs, weights, output_shape, [1, stride, stride, 1],
                            padding=padding)
        if batch_norm_params is not None:
            with slim.scopes.arg_scope([slim.ops.batch_norm], is_training=is_training,
                                trainable=trainable, restore=restore):
                outputs = slim.ops.batch_norm(deconv, **batch_norm_params)
        else:
            bias_shape = [output_shape[-1], ]
            bias_initializer = tf.constant_initializer(bias)
            biases = slim.variables.variable('biases',
                                            shape=bias_shape,
                                            initializer=bias_initializer,
                                            trainable=trainable,
                                            restore=restore)
            outputs = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if activation:
            outputs = activation(outputs)
        return outputs


# def build_generator(images):  # todo: move to kingstreet.py or parametrize
#     """
#     build_net_m: (images1, images2, is_training) -> logits
#     """
#
#     wd = 0
#
#     net = images
#
#     with tf.variable_scope('generator'):
#
#         with slim.arg_scope([slim.ops.conv2d, deconv2d], stddev=0.1, weight_decay=wd,
#                             is_training=True):
#
#             net = conv1 = slim.ops.conv2d(net, 8, [3, 3], batch_norm_params={}, scope='conv1')
#             net = pool1 = slim.ops.max_pool(net, [2, 2], scope='pool1')
#
#             net = conv2 = slim.ops.conv2d(net, 16, [3, 3], batch_norm_params={}, scope='conv2')
#             net = pool2 = slim.ops.max_pool(net, [2, 2], scope='pool2')
#
#             net = conv3 = slim.ops.conv2d(net, 32, [3, 3], batch_norm_params={}, scope='conv3')
#             net = pool3 = slim.ops.max_pool(net, [2, 2], scope='pool3')
#
#             net = conv4 = slim.ops.conv2d(net, 64, [3, 3], batch_norm_params={}, scope='conv4')
#             # net = pool4 = slim.ops.max_pool(net, [2, 2], scope='pool4')
#
#             # net = conv5 = slim.ops.conv2d(net, 128, [3, 3], batch_norm_params={}, scope='conv5')
#             # net = pool5 = slim.ops.max_pool(net, [2, 2], scope='pool5')
#
#             # print net.get_shape()
#
#             # net = deconv2d(net, [3, 3], conv4.get_shape(), batch_norm_params={}, scope='deconv5')
#             # print net.get_shape()
#
#             # net = tf.concat(3, [net, conv4], name='concat4')
#             net = deconv2d(net, [3, 3], conv3.get_shape(), batch_norm_params={}, scope='deconv4')
#             # print net.get_shape()
#
#             net = tf.concat(3, [net, conv3], name='concat3')
#             net = deconv2d(net, [3, 3], conv2.get_shape(), batch_norm_params={}, scope='deconv3')
#             # print net.get_shape()
#
#             net = tf.concat(3, [net, conv2], name='concat2')
#             net = deconv2d(net, [3, 3], images.get_shape(), scope='deconv2')
#             # print net.get_shape()
#
#             # net = tf.concat(3, [net, pool1], name='concat1')
#             # net = deconv2d(net, [3, 3], images.get_shape(), activation=None, scope='deconv1')
#             # print net.get_shape()
#
#     return net
#
#
# def build_discriminator(images, reuse=False):  # todo: move to kingstreet.py or parametrize
#     wd = 0
#
#     net = images
#
#     with tf.variable_scope('discriminator'):
#
#         with slim.arg_scope([slim.ops.conv2d], stddev=0.1, weight_decay=wd, is_training=True):
#
#             if reuse:
#                 tf.get_variable_scope().reuse_variables()
#
#             net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 8, [3, 3], batch_norm_params={}, scope='conv1')
#             net = slim.ops.max_pool(net, [2, 2], scope='pool1')
#
#             net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 16, [3, 3], batch_norm_params={}, scope='conv2')
#             net = slim.ops.max_pool(net, [2, 2], scope='pool2')
#
#             net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 32, [3, 3], batch_norm_params={}, scope='conv3')
#             net = slim.ops.max_pool(net, [2, 2], scope='pool3')
#
#             net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 64, [3, 3], batch_norm_params={}, scope='conv4')
#             net = slim.ops.max_pool(net, [2, 2], scope='pool4')
#
#             net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 128, [3, 3], batch_norm_params={}, scope='conv5')
#             net = slim.ops.max_pool(net, [2, 2], scope='pool5')
#
#             net = slim.ops.repeat_op(1, net, slim.ops.conv2d, 1, [3, 3], activation=None, scope='conv6')
#
#             net = tf.reduce_mean(net, reduction_indices=[1, 2, 3], name='reduce')
#             # net = tf.nn.sigmoid(net)
#
#     return net


def run_gan(get_data_m, get_generator, get_discriminator, batch_size, image_size,
            iter_n=1000000, init_rate=0.0001, logs_dir='logs_tmp/',
            save_dir='save_tmp/', need_load=False):
    """
    build_net_m: (t_images1, t_images2, is_trainable) -> t_logits
    get_data_m: () -> (images1, images2)
    """

    create_dir(logs_dir)
    create_dir(save_dir)
    save_file_path = os.path.join(save_dir, 'model.ckpt')

    images1 = tf.placeholder(tf.float32, [batch_size] + image_size, name='images1')  # todo: Change to variable shapes
    images2 = tf.placeholder(tf.float32, [batch_size] + image_size, name='images2')

    # Build nets
    G = get_generator(images1)

    D_true = get_discriminator(images1, images2)  # Checks true image pairs
    D_false = get_discriminator(images2, images1, reuse=True)         # Checks false image pairs
    D_gen = get_discriminator(images1, G, reuse=True)  # Checks generated image pairs

    d_vars = []
    g_vars = []
    for var in tf.trainable_variables():
        name = var.op.name
        if name.startswith('discriminator/'):
            d_vars.append(var)
        if name.startswith('generator/'):
            g_vars.append(var)

    # C = build_net_m(images1, G, is_training=False)  # Trained classifier

    dt_sum = tf.histogram_summary("dt", D_true)
    df_sum = tf.histogram_summary("df", D_false)
    dg_sum = tf.histogram_summary("dg", D_gen)
    g_sum = tf.image_summary("g", G)
    g_sum_2 = tf.image_summary("left_g", tf.concat(2, [images1, G]))

    # Build losses
    dt_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_true, tf.ones_like(D_true)))
    df_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_false, tf.zeros_like(D_false)))
    dg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_gen, tf.zeros_like(D_gen)))
    g_loss_discr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_gen, tf.ones_like(D_gen)))
    # g_loss_net = train.build_loss(C, tf.constant(1, dtype=tf.int64, shape=[BATCH_SIZE])) * 50.0
    g_loss_reg = tf.div(slim.losses.l1_loss(G - images1, weight=2e-6, scope='l1_loss'), batch_size)
    g_loss_l2im2 = tf.div(slim.losses.l2_loss(G - images2, weight=2e-6, scope='l2im2_loss'), batch_size)

    dt_loss_sum = tf.scalar_summary("dt_loss", dt_loss)
    df_loss_sum = tf.scalar_summary("df_loss", df_loss)
    dg_loss_sum = tf.scalar_summary("dg_loss", dg_loss)
    g_loss_discr_sum = tf.scalar_summary("g_loss_discr", g_loss_discr)
    # g_loss_net_sum = tf.scalar_summary("g_loss_net", g_loss_net)
    g_loss_reg_sum = tf.scalar_summary("g_loss_reg", g_loss_reg)
    g_loss_l2im2_sum = tf.scalar_summary("g_loss_l2im2", g_loss_l2im2)

    # d_loss = dt_loss * 3.0 + df_loss * 3.0 + dg_loss * 2.0
    d_loss = dt_loss + dg_loss
    # g_loss = g_loss_discr + g_loss_reg + g_loss_l2im2
    # g_loss = g_loss_discr + g_loss_l2im2
    # g_loss = g_loss_reg
    g_loss = g_loss_l2im2

    d_loss_sum = tf.scalar_summary("d_loss", d_loss)
    g_loss_sum = tf.scalar_summary("g_loss", g_loss)

    # Build optimizers
    g_opt = tf.train.AdamOptimizer(init_rate, name='train_G')
    g_grads = g_opt.compute_gradients(g_loss, var_list=g_vars)
    d_opt = tf.train.AdamOptimizer(init_rate, name='train_D')
    d_grads = d_opt.compute_gradients(d_loss, var_list=d_vars)

    d_apply_grad = d_opt.apply_gradients(d_grads)
    g_apply_grad = g_opt.apply_gradients(g_grads)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    for grad, var in d_grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    for grad, var in g_grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    step = slim.variables.variable('step_ref', shape=[], initializer=tf.constant_initializer(0), dtype=tf.int64,
                                   trainable=False)
    step = tf.assign(step, tf.add(step, 1), name='global_step')

    merged_summaries = tf.merge_all_summaries()

    sess = tf.get_default_session()

    # old_variables = []
    # for var in tf.get_collection(slim.variables.VARIABLES_TO_RESTORE):
    #     if (var.op.name.startswith('discriminator') or
    #             var.op.name.startswith('generator') or
    #             var.op.name == 'step_ref'):
    #         pass
    #     else:
    #         old_variables.append(var)

    saver = tf.train.Saver(tf.get_collection(slim.variables.VARIABLES_TO_RESTORE))
    # tmp_saver = tf.train.Saver(g_vars)
    # old_saver = tf.train.Saver(old_variables)
    writer = tf.train.SummaryWriter(logs_dir, sess.graph, flush_secs=30)

    sess.run(tf.initialize_all_variables())

    tf.train.start_queue_runners()

    if need_load:
        train.load_model(saver, sess, save_file_path)
    else:
        pass
        # train.load_model(old_saver, sess, 'save_kingstreet/model.ckpt') # todo: remove kingstreet

    my_print("Starting...\n")

    for i in range(0, iter_n):
        im1, im2 = get_data_m()
        feed = {
            images1: im1,
            images2: im2
        }
        st = step.eval()
        g_loss_val = 0.0
        for j in range(2):
            _, g_loss_val = sess.run([g_apply_grad, g_loss], feed)
        if g_loss_val < 5 and False:
            d_apply_grad.run(feed)
        if st % 10 == 0:
            summary_str = merged_summaries.eval(feed)
            my_print('Current step: %i\n' % st)
            writer.add_summary(summary_str, st)

        if st % 100 == 0:
            train.save_model(saver, sess, save_file_path)
