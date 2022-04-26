import tensorflow as tf

from cogsci_2019_concept_learning.utils.utils_gen import set_func_name


@set_func_name('lrelu')
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        #return tf.maximum(leak*x, x)
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * tf.abs(x)


@set_func_name('prelu')
def prelu(x, name="prelu"):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alphas * (x - tf.abs(x)) * 0.5
    return pos + neg


@set_func_name('relu')
def relu(x, name="relu"):
    return tf.nn.relu(x, name=name)


@set_func_name('sigmoid')
def sigmoid(x, name="sigmoid"):
    return tf.nn.sigmoid(x, name=name)


@set_func_name('tanh')
def tanh(x, name="tanh"):
    return tf.nn.tanh(x, name=name)
