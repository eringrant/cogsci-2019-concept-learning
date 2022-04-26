import numpy as np
import tensorflow as tf

from cogsci_2019_concept_learning.core.arch import conv, convfc, feedforward
from cogsci_2019_concept_learning.core.ops import log_softmax
from cogsci_2019_concept_learning.core.norm import virtual_batch_norm


class Loss(object):

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Abstract method")


class BinaryClassificationLoss(Loss):

    def __call__(self, outputs, labels):
        float_labels = tf.cast(labels, tf.float32)
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=float_labels, logits=outputs)


class MultiClassClassificationLoss(Loss):

    def __call__(self, outputs, labels):
        return tf.reduce_mean(-tf.reduce_sum(tf.cast(labels, tf.float32) * log_softmax(outputs), reduction_indices=[1]))


class NormalDistributionLoss(Loss):

    def __call__(self, outputs, labels, eps=1e-20):
        dist, features = outputs
        labels = tf.squeeze(labels)
        #features = tf.Print(features, [features, dist.log_prob(features)])#, summarize=10000)
        return - tf.reduce_sum(dist.log_prob(features))
        p = dist.prob(features)
        p = tf.Print(p, [tf.log(tf.divide(p + eps, 1 - p + eps)), labels], summarize=10000)
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=tf.log(tf.divide(p + eps, 1 - p + eps)))


class Accuracy(object):

    def __call__(self, *args, **kwargs): raise NotImplementedError("Abstract method")


class MultiClassPerClassClassificationAccuracy(object):

    def __call__(self, outputs, labels):
        num_classes = labels.get_shape().as_list()[-1]

        predictions = tf.squeeze(tf.argmax(tf.nn.softmax(outputs), axis=-1), squeeze_dims=[-1])
        labels = tf.squeeze(tf.argmax(labels, axis=-1), squeeze_dims=[-1])

        conf = tf.contrib.metrics.confusion_matrix(labels, predictions, num_classes=num_classes, dtype=tf.float32)
        return tf.diag_part(conf) / tf.reduce_sum(conf, axis=-1)
  

class MultiClassClassificationAccuracy(object):

    def __call__(self, outputs, labels):
        return tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputs), axis=-1), tf.argmax(labels, axis=-1))


class NormalDistributionAccuracy(object):

    def __call__(self, outputs, labels):
        dist, features = outputs
        p = dist.prob(features)
        correct_prediction = tf.equal(tf.round(p), labels)
        return tf.reduce_mean(tf.cast(correct_prediction, "float"))


class BinaryPerClassClassificationAccuracy(object):

    def __call__(self, outputs, labels):
        predictions = tf.squeeze(tf.cast(tf.round(tf.nn.sigmoid(outputs)), tf.int32), squeeze_dims=[-1])
        labels = tf.squeeze(tf.cast(labels, tf.int32), squeeze_dims=[-1])

        conf = tf.contrib.metrics.confusion_matrix(labels, predictions, num_classes=2, dtype=tf.float32)
        return tf.divide(tf.diag_part(conf), tf.reduce_sum(conf, axis=-1))


class BinaryClassificationAccuracy(object):

    def __call__(self, outputs, labels):
        outputs = tf.nn.sigmoid(outputs)
        correct_prediction = tf.equal(tf.round(outputs), labels)
        return tf.reduce_mean(tf.cast(correct_prediction, "float"))


class Network(object):

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Abstract method")
class ConvNet(Network):

    def __init__(self, init, nonlin, norm, output_size, size_reference_batch=None):
        self.init = init
        self.nonlin = nonlin
        self.norm = norm
        self.output_size = output_size
        self.size_reference_batch = size_reference_batch

    @property
    def dim_hidden(self):
        raise NotImplementedError("Abstract method")

    @property
    def pool(self):
        raise NotImplementedError("Abstract method")

    def __call__(self, inputs, train, weights=None):

        conv_output =  conv(inputs,
                            nonlin=self.nonlin,
                            dim_hidden=self.dim_hidden,
                            norm=self.norm,
                            pool=self.pool,
                            initializer_w=self.init,
                            output_size=self.output_size,
                            train=train,
                            weights=weights,
                            size_reference_batch=self.size_reference_batch)

        if self.norm == virtual_batch_norm:
            assert self.size_reference_batch is not None
            # Split off the reference batch
            conv_output, _ = tf.split(conv_output, [conv_output.get_shape().as_list()[0] - self.size_reference_batch, self.size_reference_batch])

        return conv_output


class SmallConvNet(ConvNet):

    @property
    def dim_hidden(self):
        return [32, 32, 32, 32]

    @property
    def pool(self):
        return None


class LargeConvNet(ConvNet):

    @property
    def dim_hidden(self):
        return [64, 64, 64, 64]

    @property
    def pool(self):
        return None


class SmallMaxPoolConvNet(ConvNet):

    @property
    def dim_hidden(self):
        return [32, 32, 32, 32]

    @property
    def pool(self):
        return 'MAX'


class LargeMaxPoolConvNet(ConvNet):

    @property
    def dim_hidden(self):
        return [64, 64, 64, 64]

    @property
    def pool(self):
        return 'MAX'


class FullyConnectedNetwork(Network):

    def __init__(self, init, nonlin, norm, output_size):
        self.init = init
        self.nonlin = nonlin
        self.norm = norm
        self.output_size = output_size

    @property
    def dim_hidden(self):
        raise NotImplementedError("Abstract method")

    def __call__(self, inputs, train, weights=None):

        if len(inputs.shape) > 2:
            inputs = tf.reshape(inputs, [-1, np.prod(inputs.get_shape().as_list()[1:])])

        net_output =  feedforward(inputs,
                                  dim_hidden=self.dim_hidden,
                                  nonlin=self.nonlin,
                                  norm=self.norm,
                                  initializer_w=self.init,
                                  output_size=self.output_size,
                                  train=train,
                                  weights=weights)

        if self.norm == virtual_batch_norm:
            # Split off the reference batch
            net_output, _ = tf.split(net_output, [net_output.get_shape().as_list()[0] - self.size_reference_batch, self.size_reference_batch])

        return net_output



class SmallOneLayerFullyConnectedNetwork(FullyConnectedNetwork):

    @property
    def dim_hidden(self):
        return [32]


class LargeOneLayerFullyConnectedNetwork(FullyConnectedNetwork):

    @property
    def dim_hidden(self):
        return [64]


class SmallTwoLayerFullyConnectedNetwork(FullyConnectedNetwork):

    @property
    def dim_hidden(self):
        return [32]


class LargeTwoLayerFullyConnectedNetwork(FullyConnectedNetwork):

    @property
    def dim_hidden(self):
        return [64]



class GenerativeConvNet(Network):

    def __init__(self, init, nonlin, norm, output_size, size_reference_batch=None):
        self.init = init
        self.nonlin = nonlin
        self.norm = norm
        self.output_size = output_size
        self.size_reference_batch = size_reference_batch

        self.dim_mu = 512

    @property
    def dim_hidden(self):
        raise NotImplementedError("Abstract method")

    @property
    def pool(self):
        raise NotImplementedError("Abstract method")

    def __call__(self, inputs, train, weights=None):

        conv_output =  conv(inputs,
                            nonlin=self.nonlin,
                            dim_hidden=self.dim_hidden,
                            norm=self.norm,
                            pool=self.pool,
                            initializer_w=self.init,
                            output_size=self.dim_mu+self.dim_mu+self.dim_mu,
                            train=train,
                            weights=weights,
                            size_reference_batch=self.size_reference_batch)

        if self.norm == virtual_batch_norm:
            assert self.size_reference_batch is not None
            # Split off the reference batch
            conv_output, _ = tf.split(conv_output, [conv_output.get_shape().as_list()[0] - self.size_reference_batch, self.size_reference_batch])

        features, mu, matrix = tf.split(conv_output, [self.dim_mu, self.dim_mu, self.dim_mu], 1)

        # Get a trainable Cholesky factor.
        #matrix = tf.reshape(matrix, [conv_output.get_shape().as_list()[0], self.dim_mu, self.dim_mu])
        #chol = tf.contrib.distributions.matrix_diag_transform(matrix, transform=tf.nn.softplus)

        #dist = tf.contrib.distributions.MultivariateNormalDiag(mu, matrix)
        dist = tf.contrib.distributions.MultivariateNormalDiag(tf.zeros(self.dim_mu), tf.ones(self.dim_mu))

        probs = fd

        return dist, features
        return - tf.reduce_sum(dist.log_prob(features))
        p = dist.prob(features)


class LargeGenerativeConvNet(GenerativeConvNet):

    @property
    def dim_hidden(self):
        return [64, 64, 64, 64]

    @property
    def pool(self):
        return None

