import os
import numpy as np
import tensorflow as tf

from cogsci_2019_concept_learning.core.ops import conv2d, linear


def feedforward(inputs, nonlin, norm, output_size, dim_hidden=[20], weights=None, train=False, initializer_w=None):

    # Normalization
    def norm_fn(x, name):
        return norm(x, training=train, name=name)

    initializer_w = initializer_w or tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32, name="fc_initializer_weights")
    initializer_b = tf.constant_initializer(0, dtype=tf.float32)

    for i, dim in enumerate(dim_hidden):
        layer_in = linear(inputs, dim, initializer_w=initializer_w, initializer_bias=initializer_b, name='fc%d' % i, weights=weights)
        inputs = nonlin(norm_fn(layer_in, name='norm_%d' % i))

    return linear(inputs, output_size, initializer_w=tf.random_normal_initializer(1e-4), initializer_bias=initializer_b, name='fc%d' % (i + 1), weights=weights)


def conv(inputs, nonlin, norm, output_size, dim_hidden=[32, 32, 32, 32], pool=None, train=False, weights=None, initializer_w=None, size_reference_batch=None):

    # Pooling
    if pool is None:
        s_h = 2; s_w = 2
        def pool_fn(x, name): return x
    elif pool == 'MAX':
        s_h = 1; s_w = 1
        def pool_fn(x, name): return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name=name)
    else:
        raise NotImplementedError

    # Normalization
    def norm_fn(x, name):
        return norm(x, weights=weights, training=train, name=name, size_reference_batch=size_reference_batch)

    # Kernel size
    k_h = 3; k_w = 3

    initializer_w = initializer_w or tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32)

    for i, dim in enumerate(dim_hidden):
        layer_in = conv2d(inputs, dim_output=dim, k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w, initializer_w=initializer_w, name='conv%d' % i, weights=weights)
        inputs = pool_fn(nonlin(norm_fn(layer_in, name='norm%d' % i)), name='pool%d' % i)

    conv_flat = tf.reshape(inputs, [-1, np.prod([int(dim) for dim in inputs.get_shape()[1:]])])

    if output_size is not None:
        output = linear(conv_flat, output_size, initializer_w=tf.random_normal_initializer(1e-4), name='fc%d' % (i + 1), weights=weights)
    else:
        output = conv_flat

    return output


def convfc(inputs, nonlin, norm, output_size, dim_hidden=[32, 32, 32, 32], pool=None, train=False, weights=None, initializer_w=None):

    # Pooling
    if pool is None:
        s_h = 2; s_w = 2
        def pool_fn(x, name): return x
    elif pool == 'MAX':
        s_h = 1; s_w = 1
        def pool_fn(x, name): return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name=name)
    else:
        raise NotImplementedError

    # Normalization
    def norm_fn(x, name):
        return norm(x, is_train=train, name=name)

    # Kernel size
    k_h = 3; k_w = 3

    initializer_w = initializer_w or tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32)

    for i, dim in enumerate(dim_hidden):
        layer_in = conv2d(inputs, dim_output=dim, k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w, initializer_w=initializer_w, name='conv%d' % i, weights=weights)
        inputs = pool_fn(nonlin(norm_fn(layer_in, name='norm%d' % i)), name='pool%d' % i)

    conv_flat = tf.reshape(inputs, [-1, np.prod([int(dim) for dim in inputs.get_shape()[1:]])])
    fc = linear(conv_flat, 200,         initializer_w=initializer_w,                      name='fc%d' % (i + 1), weights=weights)
    fc = linear(fc,        output_size, initializer_w=tf.random_normal_initializer(1e-4), name='fc%d' % (i + 2), weights=weights)

    return fc


def alexnet_pretrained(inputs, weights=None):

    net_data = np.load(os.path.join('weights', 'bvlc_alexnet.npy'), encoding="latin1").item()

    # conv1
    k_h = 11; k_w = 11; s_h = 4; s_w = 4; c_o = 96; group = 1
    conv1_in = conv2d(inputs, dim_output=c_o,
                      k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w, group=group,
                      initializer_w=tf.constant_initializer(net_data["conv1"][0]),
                      initializer_bias=tf.constant_initializer(net_data["conv1"][1]),
                      name="conv1", weights=weights)
    conv1 = tf.nn.relu(conv1_in)

    # lrn1
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name='lrn1')

    # pool1
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool1')

    # conv2
    k_h = 5; k_w = 5; s_h = 1; s_w = 1; c_o = 256; group = 2
    conv2_in = conv2d(maxpool1, dim_output=c_o,
                      k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w, group=group,
                      initializer_w=tf.constant_initializer(net_data["conv2"][0]),
                      initializer_bias=tf.constant_initializer(net_data["conv2"][1]),
                      name="conv2", weights=weights)
    conv2 = tf.nn.relu(conv2_in)

    # lrn2
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name='lrn2')

    # pool2
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool2')

    # conv3
    k_h = 3; k_w = 3; s_h = 1; s_w = 1; c_o = 384; group = 1
    conv3_in = conv2d(maxpool2, dim_output=c_o,
                      k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w, group=group,
                      initializer_w=tf.constant_initializer(net_data["conv3"][0]),
                      initializer_bias=tf.constant_initializer(net_data["conv3"][1]),
                      name="conv3", weights=weights)
    conv3 = tf.nn.relu(conv3_in)

    # conv4
    k_h = 3; k_w = 3; s_h = 1; s_w = 1; c_o = 384; group = 2
    conv4_in = conv2d(conv3, dim_output=c_o,
                      k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w, group=group,
                      initializer_w=tf.constant_initializer(net_data["conv4"][0]),
                      initializer_bias=tf.constant_initializer(net_data["conv4"][1]),
                      name="conv4", weights=weights)
    conv4 = tf.nn.relu(conv4_in)

    # conv5
    k_h = 3; k_w = 3; s_h = 1; s_w = 1; c_o = 256; group = 2
    conv5_in = conv2d(conv4, dim_output=c_o,
                      k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w, group=group,
                      initializer_w=tf.constant_initializer(net_data["conv5"][0]),
                      initializer_bias=tf.constant_initializer(net_data["conv5"][1]),
                      name="conv5", weights=weights)
    conv5 = tf.nn.relu(conv5_in)

    # pool5
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool5')

    # fc6
    output_size = 4096
    fc6_in = linear(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]),
                    output_size,
                    initializer_w=tf.constant_initializer(net_data["fc6"][0]),
                    initializer_bias=tf.constant_initializer(net_data["fc6"][1]),
                    name='fc6', weights=weights)
    fc6 = tf.nn.relu(fc6_in)

    # fc7
    output_size = 4096
    fc7 = linear(fc6,
                 output_size,
                 initializer_w=tf.constant_initializer(net_data["fc7"][0]),
                 initializer_bias=tf.constant_initializer(net_data["fc7"][1]),
                 name='fc7')

    return fc7
