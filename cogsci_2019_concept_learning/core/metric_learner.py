import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops.nn_ops import max_pool

from cogsci_2019_concept_learning.core.model import *
from cogsci_2019_concept_learning.utils.utils_gen import br, log_function_call
from cogsci_2019_concept_learning.utils.utils_tf import cosine_distance, euclidean_distance, print_numeric_tensor


class MetricLearner(object):

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
        self.num_classes = num_classes

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

    def accuracy(self, pred, gt):
        if self.num_classes > 1:
            correct_prediction = tf.equal(tf.argmax(pred, -1), tf.argmax(gt, -1))
        else:
            correct_prediction = tf.equal(tf.round(pred), gt)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def loss_func(self, pred, gt, eps=1.e-10):
        if self.num_classes > 1:
            xent = -tf.reduce_sum(gt * tf.log(pred + eps), reduction_indices=[1])
        else:
            xent  = -tf.reduce_sum(gt * tf.log(pred + eps), reduction_indices=[1])
            xent += -tf.reduce_sum((1 - gt) * tf.log(1 - pred + eps), reduction_indices=[1])
        return tf.reduce_mean(xent)

    def f(self, support, test, train):
        raise NotImplementedError("Abstract method")

    def g(self, support, labels, train):
        raise NotImplementedError("Abstract method")

    def compute_similarity(self, *args, **kwargs):
        raise NotImplementedError("Abstract method")

    def classify(self, similarities, support_set_labels, train):
        raise NotImplementedError("Abstract method")

    def construct_weights(self, scope, reuse):

        with tf.variable_scope(scope, reuse=reuse):
            train_batch_size = self.update_training_batch_size
            valid_batch_size = self.update_validation_batch_size

            if self.norm == virtual_batch_norm:
                train_batch_size += self.size_reference_batch
                valid_batch_size += self.size_reference_batch

            dummy_train_input = tf.zeros([train_batch_size] + self.input_shape, name="dummy_training_input")
            dummy_valid_input = tf.zeros([valid_batch_size] + self.input_shape, name="dummy_validation_input")

            self.f(dummy_train_input, dummy_valid_input, False)
            # g network just uses the same weights for now

    def build_model(self, reuse, input_tensors, train, prefix, return_outputs=False, test=False, scope='model'):

        self.construct_weights(scope, reuse=reuse)

        inputa = input_tensors['training data']
        labela = input_tensors['training labels']
        inputb = input_tensors['validation data']
        labelb = input_tensors['validation labels']

        # Switch to one-hot label encoding
        if labela.get_shape().as_list()[-1] == 1 and self.num_classes > 1:
            labela = tf.squeeze(tf.one_hot(tf.cast(labela, tf.int32), depth=self.num_classes, axis=-1), squeeze_dims=[-2])
            labelb = tf.squeeze(tf.one_hot(tf.cast(labelb, tf.int32), depth=self.num_classes, axis=-1), squeeze_dims=[-2])

        with tf.variable_scope(scope, reuse=True):

            @log_function_call("map of batch meta-learn function")
            def batch_metalearn(inp):
                """a for train, b for test

                Arguments:
                    inputa: A [meta_batch_size, num_classes*samples_per_class, dim_input] Tensor.
                    labela: A [meta_batch_size, num_classes*samples_per_class, num_classes] Tensor.
                    inputb: A [meta_batch_size, num_classes*samples_per_class, dim_input] Tensor.
                    labelb: A [meta_batch_size, num_classes*samples_per_class, num_classes] Tensor.
                """
                local_inputa, local_inputb, local_labela, local_labelb = inp

                # Produce embeddings for test and support set images
                encoded_test                    = self.f(local_inputa, local_inputb, train=train)
                encoded_support, support_labels = self.g(local_inputa, local_labela, train=train)

                # Compute distances
                similarities = self.compute_similarity(support_set=encoded_support,
                                                       test_set=encoded_test,
                                                       train=train)

                # Predict labels
                preds = self.classify(similarities,
                                      support_set_labels=support_labels,
                                      train=train)

                local_accuracy = self.accuracy(preds, local_labelb)
                local_loss = self.loss_func(preds, local_labelb)

                return local_accuracy, local_loss

            out_dtype = (tf.float32, tf.float32)
            meta_batch_size = int(inputa.get_shape().as_list()[0])
            accuracies, losses = tf.map_fn(batch_metalearn, elems=(inputa, inputb, labela, labelb), dtype=out_dtype, parallel_iterations=meta_batch_size, name="batch_metalearn")

            total_loss     = tf.reduce_mean(losses)
            total_accuracy = tf.reduce_mean(accuracies)

            tensor_outputs = {
                'total loss':     total_loss,
                'total accuracy': total_accuracy,
            }

        tf.summary.scalar('%s/loss' % prefix,     total_loss)
        tf.summary.scalar('%s/accuracy' % prefix, total_accuracy)

        if train:

            optimizer = self.optimizer(self.step_size)
            train_gvs = optimizer.compute_gradients(total_loss)

            if self.clip_norm:
                train_gvs = [(tf.clip_by_norm(grad, self.clip_norm, name="grad_clip_%s" % var.name.replace(prefix + "/", "").replace("/", "_").replace(":0", "")), var) for grad, var in train_gvs]

            train_op = optimizer.apply_gradients(train_gvs)
            tensor_outputs['train_op'] = train_op

            # Summaries
            for grad, var in train_gvs:
                var_name = '/'.join(var.name.split('/')[-2:])
                tf.summary.histogram('train_variables/%s'      % var_name.replace(":", "_"), var)
                tf.summary.histogram('train_gradients/%s' % var_name.replace(":", "_"), grad)

        return tensor_outputs


class CosineDistanceMetricLearner(MetricLearner):

    def compute_similarity(self, support_set, test_set, train):

        similarities = []
        for im in tf.unstack(test_set, axis=0):
            im = tf.expand_dims(im, 0)
            similarities += [1 - cosine_distance(support_set, im)]

        return tf.squeeze(tf.stack(similarities), squeeze_dims=[-1])


class EuclideanDistanceMetricLearner(MetricLearner):

    def compute_similarity(self, support_set, test_set, train):

        similarities = []
        for im in tf.unstack(test_set, axis=0):
            im = tf.expand_dims(im, 0)
            similarities += [-euclidean_distance(support_set, im)]

        return tf.squeeze(tf.stack(similarities), squeeze_dims=[-1])


class MatchingNetwork(MetricLearner):

    def __init__(self, *args, **kwargs):
        assert kwargs['num_classes'] > 1, "Matching networks cannot be applied to a unary classification task"
        super().__init__(*args, **kwargs)

    def classify(self, similarities, support_set_labels, train):
        softmax_similarities = tf.nn.softmax(similarities)
        preds = tf.matmul(softmax_similarities, support_set_labels)
        return preds

    def f(self, support, test, train):
        return self.f_network(test, train)

    def g(self, support, labels, train):
        return self.g_network(support, train), labels


class PrototypicalNetwork(MetricLearner):

    def classify(self, similarities, support_set_labels, train):
        if self.num_classes > 1:
            softmax_similarities = tf.nn.softmax(similarities)
            preds = tf.matmul(softmax_similarities, support_set_labels)
        else:
            assert similarities.get_shape().as_list()[-1] == 1
            preds = tf.nn.sigmoid(similarities)
        return preds

    def f(self, support, test, train):
        return self.f_network(test, train)

    def g(self, support, labels, train):
        labels = tf.argmax(labels, axis=-1)
        encoded_support = self.g_network(support, train)

        if self.num_classes > 1:
            prototypes = []
            for lbl in range(self.num_classes):
                prototypes += [tf.reduce_mean(tf.boolean_mask(encoded_support, tf.equal(labels, lbl)), axis=0)]
            prototype_features = tf.stack(prototypes)
            prototype_labels = tf.one_hot(tf.range(self.num_classes), depth=self.num_classes)

        else:
            prototype_features = tf.stack([
                tf.reduce_mean(encoded_support, axis=0),
            ])
            prototype_labels = tf.constant([1])

        return prototype_features, prototype_labels


class SmallConvCosineMatchingNetwork(CosineDistanceMetricLearner, MatchingNetwork):
    name = 'small_conv_cosmatchnet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = SmallConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = SmallConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class LargeConvCosineMatchingNetwork(CosineDistanceMetricLearner, MatchingNetwork):
    name = 'large_conv_cosmatchnet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = LargeConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = LargeConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class SmallConvMaxPoolCosineMatchingNetwork(CosineDistanceMetricLearner, MatchingNetwork):
    name = 'small_convmaxpool_cosmatchnet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = SmallMaxPoolConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = SmallMaxPoolConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class LargeConvMaxPoolCosineMatchingNetwork(CosineDistanceMetricLearner, MatchingNetwork):
    name = 'large_convmaxpool_cosmatchnet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = LargeMaxPoolConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = LargeMaxPoolConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class SmallOneLayerFullyConnectedCosineMatchingNetwork(CosineDistanceMetricLearner, MatchingNetwork):
    name = 'small_onelayerfc_cosmatchnet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = SmallOneLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = SmallOneLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class LargeOneLayerFullyConnectedCosineMatchingNetwork(CosineDistanceMetricLearner, MatchingNetwork):
    name = 'large_onelayerfc_cosmatchnet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = LargeOneLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = LargeOneLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class SmallTwoLayerFullyConnectedCosineMatchingNetwork(CosineDistanceMetricLearner, MatchingNetwork):
    name = 'small_twolayerfc_cosmatchnet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = SmallTwoLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = SmallTwoLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class LargeTwoLayerFullyConnectedCosineMatchingNetwork(CosineDistanceMetricLearner, MatchingNetwork):
    name = 'large_twolayerfc_cosmatchnet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = LargeTwoLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = LargeTwoLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class SmallConvEuclideanMatchingNetwork(EuclideanDistanceMetricLearner, MatchingNetwork):
    name = 'small_conv_euclidmatchnet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = SmallConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = SmallConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class LargeConvEuclideanMatchingNetwork(EuclideanDistanceMetricLearner, MatchingNetwork):
    name = 'large_conv_euclidmatchnet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = LargeConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = LargeConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class SmallConvMaxPoolEuclideanMatchingNetwork(EuclideanDistanceMetricLearner, MatchingNetwork):
    name = 'small_convmaxpool_euclidmatchnet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = SmallMaxPoolConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = SmallMaxPoolConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class LargeConvMaxPoolEuclideanMatchingNetwork(EuclideanDistanceMetricLearner, MatchingNetwork):
    name = 'large_convmaxpool_euclidmatchnet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = LargeMaxPoolConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = LargeMaxPoolConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class SmallOneLayerFullyConnectedEuclideanMatchingNetwork(EuclideanDistanceMetricLearner, MatchingNetwork):
    name = 'small_onelayerfc_euclidmatchnet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = SmallOneLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = SmallOneLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class LargeOneLayerFullyConnectedEuclideanMatchingNetwork(EuclideanDistanceMetricLearner, MatchingNetwork):
    name = 'large_onelayerfc_euclidmatchnet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = LargeOneLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = LargeOneLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class SmallTwoLayerFullyConnectedEuclideanMatchingNetwork(EuclideanDistanceMetricLearner, MatchingNetwork):
    name = 'small_twolayerfc_euclidmatchnet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = SmallTwoLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = SmallTwoLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class LargeTwoLayerFullyConnectedEuclideanMatchingNetwork(EuclideanDistanceMetricLearner, MatchingNetwork):
    name = 'large_twolayerfc_euclidmatchnet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = LargeTwoLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = LargeTwoLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class SmallConvCosinePrototypicalNetwork(CosineDistanceMetricLearner, PrototypicalNetwork):
    name = 'small_conv_cosprotonet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = SmallConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = SmallConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class LargeConvCosinePrototypicalNetwork(CosineDistanceMetricLearner, PrototypicalNetwork):
    name = 'large_conv_cosprotonet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = LargeConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = LargeConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class SmallConvMaxPoolCosinePrototypicalNetwork(CosineDistanceMetricLearner, PrototypicalNetwork):
    name = 'small_convmaxpool_cosprotonet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = SmallMaxPoolConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = SmallMaxPoolConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class LargeConvMaxPoolCosinePrototypicalNetwork(CosineDistanceMetricLearner, PrototypicalNetwork):
    name = 'large_convmaxpool_cosprotonet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = LargeMaxPoolConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = LargeMaxPoolConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class SmallOneLayerFullyConnectedCosinePrototypicalNetwork(CosineDistanceMetricLearner, PrototypicalNetwork):
    name = 'small_onelayerfc_cosprotonet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = SmallOneLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = SmallOneLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class LargeOneLayerFullyConnectedCosinePrototypicalNetwork(CosineDistanceMetricLearner, PrototypicalNetwork):
    name = 'large_onelayerfc_cosprotonet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = LargeOneLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = LargeOneLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class SmallTwoLayerFullyConnectedCosinePrototypicalNetwork(CosineDistanceMetricLearner, PrototypicalNetwork):
    name = 'small_twolayerfc_cosprotonet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = SmallTwoLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = SmallTwoLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class LargeTwoLayerFullyConnectedCosinePrototypicalNetwork(CosineDistanceMetricLearner, PrototypicalNetwork):
    name = 'large_twolayerfc_cosprotonet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = LargeTwoLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = LargeTwoLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class SmallConvEuclideanPrototypicalNetwork(EuclideanDistanceMetricLearner, PrototypicalNetwork):
    name = 'small_conv_euclidprotonet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = SmallConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = SmallConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class LargeConvEuclideanPrototypicalNetwork(EuclideanDistanceMetricLearner, PrototypicalNetwork):
    name = 'large_conv_euclidprotonet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = LargeConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = LargeConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class SmallConvMaxPoolEuclideanPrototypicalNetwork(EuclideanDistanceMetricLearner, PrototypicalNetwork):
    name = 'small_convmaxpool_euclidprotonet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = SmallMaxPoolConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = SmallMaxPoolConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class LargeConvMaxPoolEuclideanPrototypicalNetwork(EuclideanDistanceMetricLearner, PrototypicalNetwork):
    name = 'large_convmaxpool_euclidprotonet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = LargeMaxPoolConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = LargeMaxPoolConvNet(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class SmallOneLayerFullyConnectedEuclideanPrototypicalNetwork(EuclideanDistanceMetricLearner, PrototypicalNetwork):
    name = 'small_onelayerfc_euclidprotonet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = SmallOneLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = SmallOneLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class LargeOneLayerFullyConnectedEuclideanPrototypicalNetwork(EuclideanDistanceMetricLearner, PrototypicalNetwork):
    name = 'large_onelayerfc_euclidprotonet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = LargeOneLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = LargeOneLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class SmallTwoLayerFullyConnectedEuclideanPrototypicalNetwork(EuclideanDistanceMetricLearner, PrototypicalNetwork):
    name = 'small_twolayerfc_euclidprotonet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = SmallTwoLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        self.g_network = SmallTwoLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)


class LargeTwoLayerFullyConnectedEuclideanPrototypicalNetwork(EuclideanDistanceMetricLearner, PrototypicalNetwork):
    name = 'large_twolayerfc_euclidprotonet'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_network = LargeTwoLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
        gelf.g_network = LargeTwoLayerFullyConnectedNetwork(self.init, self.nonlin, self.norm, None, self.size_reference_batch)
