import tensorflow as tf
from cogsci_2019_concept_learning.utils.utils_gen import is_sequence 

def conv2d(inputs, dim_output,
           k_h=5, k_w=5, s_h=2, s_w=2,
           padding='SAME', group=1,
           initializer_w=tf.truncated_normal_initializer(stddev=0.02),
           initializer_bias=tf.random_normal_initializer(stddev=0.02),
           name="conv2d", weights=None):
    """TODO

    Args:
        TODO

    Returns:
        TODO
    """
    dim_input = inputs.get_shape()[-1]
    assert dim_input % group == 0, "Input dimension is not divisible by group: %d \% %d != 0" % (dim_input, group)
    assert dim_output % group == 0, "Output dimension is not divisible by group: %d \% %d != 0" % (dim_output, group)

    with tf.variable_scope(name):

        def convolve(i, k):
            return tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

        if weights:
            kernel = weights['%s/weights' % name]
        else:
            kernel = tf.get_variable('weights', [k_h, k_w, dim_input // group, dim_output], initializer=initializer_w)

        if group == 1:
            conv = convolve(inputs, kernel)

        else:
            input_groups = tf.split(3, group, inputs)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            conv = tf.concat(3, output_groups)

        if weights:
            bias = weights['%s/bias' % name]
        else:
            bias = tf.get_variable('bias', [dim_output], initializer=initializer_bias)

        result = tf.nn.bias_add(conv, bias)

    return result


def deconv2d(inputs, dim_output,
             k_h=5, k_w=5, s_h=2, s_w=2,
             init_stddev=0.02, init_bias=0.,
             initializer_w=tf.random_normal_initializer,
             initializer_bias=tf.random_normal_initializer,
             name="deconv2d"):
    """TODO

    Args:
        TODO

    Returns:
        TODO
    """
    if False:
        raise ValueError('TODO')
    with tf.variable_scope(name):

        dim_input = inputs.get_shape()[-1]

        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('weights', [k_h, k_w, dim_output[-1], dim_input],
                            initializer=initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(inputs, w, dim_output=dim_output,
                                        strides=[1, s_h, s_w, 1])

        biases = tf.get_variable('biases', [dim_output[-1]], initializer=initializer(init_bias))
        result = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return result


def linear(args, output_size,
           initializer_w=tf.truncated_normal_initializer(stddev=1.),
           initializer_bias=tf.constant_initializer(0.),
           name=None, weights=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
        args: a 2D Tensor or a list of 2D, batch_size x n, Tensors.
        output_size: int, second dimension of W[i].
        matrix_mean: starting mean value to initialize the matrix; 0 by default.
        matrix_std: starting standard deviation value to initialize the matrix; 0.1 by default.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "linear".

    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if not is_sequence(args):
        args = [args]

    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]
    assert all([dtype == a.dtype for a in args])

    with tf.variable_scope(name or "fc"):

        if weights:
            matrix = weights['%s/weights' % name]
        else:
            matrix = tf.get_variable("weights", [total_arg_size, output_size], dtype=dtype, initializer=initializer_w)

        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(1, args), matrix)

        if weights:
            bias = weights['%s/bias' % name]
        else:
            bias = tf.get_variable("bias", [output_size], dtype=dtype, initializer=initializer_bias)

        result = res + bias

    return result


def log_softmax(x):
    return x - log_sum_exp(x)


def log_sum_exp(x, epsilon=1.0e-12):
    max_ = tf.reduce_max(x, keep_dims=True)
    x -= max_
    return tf.squeeze(max_, [-1]) + tf.log(tf.reduce_sum(tf.exp(x), -1, keep_dims=True) + epsilon)


def sigmoid_kl_with_logits(logits, targets):
    """TODO

    Equivalent to cross entropy loss if the targets are not random vbls.

    Args:
        TODO

    Returns:
        TODO
    """
    assert isinstance(targets, float)  # broadcasts the same target value across the whole batch
    if targets in [0., 1.]:
        entropy = 0.
    else:
        # this is implemented so awkwardly because tensorflow lacks an x log x op
        entropy = - targets * np.log(targets) - (1. - targets) * np.log(1. - targets)
    return tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.ones_like(logits) * targets) - entropy


def switch(if_condition, then_expression, else_expression):
    """Switches between two operations depending on a scalar value (int or bool).
    Note that both `then_expression` and `else_expression`
    should be symbolic tensors of the *same shape*.

    Args:
        condition: scalar tensor.
        then_expression: TensorFlow operation.
        else_expression: TensorFlow operation.

    Returns:
        TODO
    """
    x_shape = copy.copy(then_expression.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),
                lambda: then_expression,
                lambda: else_expression)
    x.set_shape(x_shape)
    return x
