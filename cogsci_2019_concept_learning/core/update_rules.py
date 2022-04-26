import tensorflow as tf

from cogsci_2019_concept_learning.utils.utils_gen import set_func_name


@set_func_name('adam')
def adam(gradients, state, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    if state is None:
        t, m_prev, v_prev = tf.constant(0, dtype=tf.float32), tf.zeros_like(gradients), tf.zeros_like(gradients)
    else:
        t, m_prev, v_prev = state

    t += 1

    m = beta_1 * m_prev + (1 - beta_1) * gradients
    v = beta_2 * v_prev + (1 - beta_2) * tf.pow(gradients, 2)

    alpha = learning_rate * tf.sqrt(1  - tf.pow(beta_2, t)) / (1 -  tf.pow(beta_1, t))
    update = alpha * m / (tf.sqrt(v + epsilon))

    return -update, (t, m, v)


@set_func_name('rmsprop')
def rmsprop(gradients, state, learning_rate=0.1, decay_rate=0.9, epsilon=1e-5):
    if state is None:
        state = tf.zeros_like(gradients)
    state = decay_rate * state + (1 - decay_rate) * tf.pow(gradients, 2)
    update = learning_rate * gradients / (tf.sqrt(state + epsilon))
    return -update, state


@set_func_name('sgd')
def sgd(gradients, state, learning_rate=0.1):
    if state is None:
        state = tf.zeros_like(gradients)
    return -learning_rate * gradients, state
