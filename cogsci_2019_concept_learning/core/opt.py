import tensorflow as tf

from cogsci_2019_concept_learning.utils.utils_gen import set_func_name


@set_func_name('adam_9')
def adam_9_opt(learning_rate):
    return tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)


@set_func_name('adam_99')
def adam_99_opt(learning_rate):
    return tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.99, beta2=0.999, epsilon=1e-08)

