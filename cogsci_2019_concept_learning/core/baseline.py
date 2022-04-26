import numpy as np
import tensorflow as tf

from cogsci_2019_concept_learning.core.arch import conv, convfc, feedforward
from cogsci_2019_concept_learning.core.model import *
from cogsci_2019_concept_learning.core.ops import log_softmax

from cogsci_2019_concept_learning.utils.utils_gen import br, log_function_call
from cogsci_2019_concept_learning.utils.utils_tf import print_numeric_tensor, cosine_distance, euclidean_distance


def compute_distances(support_set, candidates):
    """For test time distance computation."""

    support_set_shape = support_set.get_shape().as_list()
    candidates_shape  = candidates.get_shape().as_list()

    assert len(support_set_shape) == 2
    assert len(candidates_shape)  == 2

    mean_support = tf.reduce_mean(support_set, 0, keep_dims=True)

    euclidean_distances = euclidean_distance(support_set, candidates)
    cosine_distances    = cosine_distance(   support_set, candidates)

    distances = {
        'euclidean_distance_to_mean': tf.squeeze(euclidean_distance(mean_support, candidates), squeeze_dims=[0]),
        'cosine_distance_to_mean':    tf.squeeze(cosine_distance(   mean_support, candidates), squeeze_dims=[0]),
        'mean_euclidean_distance':    tf.reduce_mean(euclidean_distances, reduction_indices=[0]),
        'min_euclidean_distance':     tf.reduce_min(euclidean_distances,  reduction_indices=[0]),
        'sum_euclidean_distance':     tf.reduce_sum(euclidean_distances,  reduction_indices=[0]),
        'mean_cosine_distance':       tf.reduce_mean(cosine_distances,    reduction_indices=[0]),
        'min_cosine_distance':        tf.reduce_min(cosine_distances,     reduction_indices=[0]),
        'sum_cosine_distance':        tf.reduce_sum(cosine_distances,     reduction_indices=[0]),
    }

    return (
        distances['euclidean_distance_to_mean'],
        distances['cosine_distance_to_mean'],
        distances['mean_euclidean_distance'],
        distances['min_euclidean_distance'],
        distances['sum_euclidean_distance'],
        distances['mean_cosine_distance'],
        distances['min_cosine_distance'],
        distances['sum_cosine_distance'],
    )


class BaselineLearner(object):

    def __init__(self,
            input_shape,
            num_classes,
            init,
            norm,
            nonlin,
            meta_batch_size,
            update_training_batch_size,
            update_validation_batch_size,
            meta_step_size,
            meta_clip_norm,
            meta_optimizer,
            size_reference_batch,
            *args,
            **kwargs):

        self.input_shape = input_shape
        self.num_classes = 494

        self.init = init
        self.norm = norm
        self.nonlin = nonlin

        self.meta_batch_size   = meta_batch_size
        self.update_training_batch_size = update_training_batch_size
        self.update_validation_batch_size = update_validation_batch_size
        self.size_reference_batch = size_reference_batch

        self.step_size = meta_step_size
        self.clip_norm = meta_clip_norm
        self.optimizer = meta_optimizer

    def accuracy(self, pred, gt, weights=None):
        correct_prediction = tf.equal(tf.argmax(pred, -1), tf.argmax(gt, -1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def loss_func(self, pred, gt, weights=None, eps=1.e-10):
        return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=gt)
        xent = -tf.reduce_sum(gt * tf.log(pred + eps), reduction_indices=[1])
        return tf.reduce_mean(xent)

    def build_model(self, reuse, input_tensors, train, prefix, return_outputs=False, test=False, scope='model'):

        assert self.meta_batch_size == 1

        inputa = input_tensors['training data']
        labela = input_tensors['training labels']
        inputb = input_tensors['validation data']
        labelb = input_tensors['validation labels']

        with tf.variable_scope('model', reuse=reuse):

            def batch_learn(inp):
                """TODO

                Arguments:
                    TODO
                """
                if not test:
                    local_input, local_label = inp
                else:
                    local_input = inp

                local_output = self.forward_pass(local_input, train, None)

                if not test:
                    local_loss     = self.loss_func(local_output, local_label, None)
                    local_accuracy = self.accuracy(local_output,  local_label, None)

                    return local_output, local_loss, local_accuracy

                else:
                    return local_output

            tensor_outputs = {}

            if not test:

                # Not using inner loop "training" data
                assert inputa is None
                assert labela is None

                # Collapse meta-batch index
                inputb = tf.squeeze(inputb, squeeze_dims=[0])
                labelb = tf.squeeze(labelb, squeeze_dims=[0])

                output_b, loss_b, accuracy_b = batch_learn((inputb, labelb))

                train_gvs = list(zip(tf.gradients(tf.reduce_mean(loss_b), tf.trainable_variables()), tf.trainable_variables()))

                if self.clip_norm:
                    train_gvs    = [(tf.clip_by_norm(grad, self.clip_norm), var) for grad, var in train_gvs]

                optimizer = self.optimizer(self.step_size)
                train_op = optimizer.apply_gradients(train_gvs)

                tensor_outputs['train_op'] = train_op

                # Expand meta-batch index for return value
                output_b   = tf.expand_dims(output_b,   0)
                loss_b     = tf.expand_dims(loss_b,     0)
                accuracy_b = tf.expand_dims(accuracy_b, 0)

                tensor_outputs.update({
                    'total_losses_b':      [loss_b],
                    'total_accuracies_b':  [accuracy_b],
                })

                if return_outputs:
                    tensor_outputs.update({
                        'outputs_b': [output_b],
                    })

            else:

                # Collapse meta-batch index
                inputa = tf.squeeze(inputa, squeeze_dims=[0])
                inputb = tf.squeeze(inputb, squeeze_dims=[0])

                outputa = batch_learn(inputa)
                tf.get_variable_scope().reuse_variables()
                outputb = batch_learn(inputb)

                distance_output = compute_distances(outputa, outputb)

                tensor_outputs['euclidean_distance_to_mean'] = distance_output[0]
                tensor_outputs['cosine_distance_to_mean'] = distance_output[1]
                tensor_outputs['mean_euclidean_distance'] = distance_output[2]
                tensor_outputs['min_euclidean_distance'] = distance_output[3]
                tensor_outputs['sum_euclidean_distance'] = distance_output[4]
                tensor_outputs['mean_cosine_distance'] = distance_output[5]
                tensor_outputs['min_cosine_distance'] = distance_output[6]
                tensor_outputs['sum_cosine_distance'] = distance_output[7]

            return tensor_outputs


class LargeConvBaseline(BaselineLearner):
    name = 'large_conv_baseline'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_pass = LargeConvNet(self.init, self.nonlin, self.norm, 494, self.size_reference_batch)
