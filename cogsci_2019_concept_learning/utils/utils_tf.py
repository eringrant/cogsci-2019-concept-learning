import functools
import tensorflow as tf


def cosine_distance(X, Y):
    """Compute pairwise cosine distances between X and Y."""
    X_shape = X.get_shape().as_list()
    Y_shape = Y.get_shape().as_list()
    assert X_shape[-1] == Y_shape[-1], "Tensors must share the last dimension."

    normed_X = tf.nn.l2_normalize(X, dim=-1)
    normed_Y = tf.nn.l2_normalize(Y, dim=-1)

    return 1 - tf.matmul(normed_X, normed_Y, transpose_b=True)


def euclidean_distance(X, Y):
    """Compute pairwise Euclidean distances between X and Y."""
    X_shape = X.get_shape().as_list()
    Y_shape = Y.get_shape().as_list()
    assert X_shape[-1] == Y_shape[-1], "Tensors must share the last dimension."

    expanded_X = tf.expand_dims(X, -2)
    expanded_Y = tf.expand_dims(Y, -3)

    return tf.sqrt(tf.reduce_sum(tf.squared_difference(expanded_X, expanded_Y), -1))


def print_numeric_tensor(assign_op, print_op, message):
    return tf.Print(assign_op, [tf.reduce_max(print_op), tf.reduce_mean(print_op), tf.reduce_min(print_op)],  message=message)


def assert_matching_dims(x, y, dims=None):
    if dims is None:
        assert len(x.get_shape()) == len(y.get_shape()), "Tensors are of different dimensionalities"
        assert  x.get_shape().as_list() == y.get_shape().as_list(), "Tensor dimensions do not match"
    else:
        x_shape = x.get_shape().as_list()
        y_shape = y.get_shape().as_list()
        for dim in dims:
            assert x_shape[dim] == y_shape[dim], "Tensor dimensions do not match in the %dth dimension" % dim


def avg_grads(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, v in grad_and_vars:

            assert g is not None, "Gradient for variable %s did not exist" % v.name

            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def load_state(fname, sess=tf.get_default_session()):
    saver = tf.train.Saver()
    saver.restore(sess, fname)


def save_state(fname, sess=tf.get_default_session()):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    saver = tf.train.Saver()
    saver.save(sess, fname)
