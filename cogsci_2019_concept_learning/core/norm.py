import tensorflow as tf

from cogsci_2019_concept_learning.utils.utils_gen import set_func_name


@set_func_name('batch_norm')
def batch_norm(x, weights, training, name=None):
    raise NotImplementedError("Incorporate weights")
    return tf.layers.batch_norm(x,
                                training=training,
                                #updates_collections=None,  # disabled because of severe minibatch dependence
                                scope=name)


@set_func_name('batch_renorm')
def batch_renorm(x, weights, training, name=None):
    raise NotImplementedError("Incorporate weights")
    return tf.layers.batch_norm(x,
                                training=training,
                                renorm_clipping={'rmax': 3., 'rmin': 1./3, 'dmax': 5},
                                renorm=True,
                                scope=name)


@set_func_name('layer_norm')
def layer_norm(x, weights, training, name=None):
    raise NotImplementedError("Incorporate weights")
    return tf.contrib.layers.layer_norm(x, scope=name)


@set_func_name('identity_norm')
def identity_norm(x, weights, training, size_reference_batch, name=None):
    return x


@set_func_name('virtual_batch_norm')
def virtual_batch_norm(x, weights, training, reference_batch_idx=None, name=None):

    axis = -1
    epsilon = 1e-3
    beta_initializer = tf.zeros_initializer()
    gamma_initializer = tf.ones_initializer()
    beta_constraint = None
    gamma_constraint = None

    # Assume that the reference batch always passed
    _, ref_batch = tf.split(x, [-1, size_reference_batch])
    input_shape = ref_batch.get_shape()

    # First, compute the axes along which to reduce the mean / variance,
    # as well as the broadcast shape to be used for all parameters.
    ndim = len(input_shape)
    reduction_axes = list(range(len(input_shape)))
    del reduction_axes[axis]
    broadcast_shape = [1] * len(input_shape)
    broadcast_shape[axis] = input_shape[axis].value

    # Determines whether broadcasting is needed.
    needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

    param_dim = input_shape[axis]
    if weights:
        scale = weights['%s/gamma' % name]
        offset = weights['%s/beta' % name]
    else:
        with tf.variable_scope(name):
            scale = tf.get_variable(name='gamma', shape=(param_dim,), initializer=gamma_initializer, trainable=True)
            offset = tf.get_variable(name='beta', shape=(param_dim,), initializer=beta_initializer, trainable=True)

    mean, variance = tf.nn.moments(ref_batch, reduction_axes)

    def _broadcast(v):
        if needs_broadcasting and v is not None:
            # In this case we must explicitly broadcast all parameters.
            return tf.reshape(v, broadcast_shape)
        return v

    return tf.nn.batch_normalization(x,
                                     _broadcast(mean),
                                     _broadcast(variance),
                                     _broadcast(offset),
                                     _broadcast(scale),
                                     epsilon)
