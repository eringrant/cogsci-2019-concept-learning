from itertools import repeat
import logging
from multiprocessing import Pool
from nltk.corpus import wordnet as wn
import numpy as np
import os
import pickle
import random

from cogsci_2019_concept_learning.utils.utils_gen import br, log_function_call, multi_pop, ss
from cogsci_2019_concept_learning.utils.utils_np import shuffle_in_unison, dense_to_one_hot


class Dataset(object):

    def __init__(self,
                 num_training_samples,
                 num_validation_samples,
                 meta_batch_size,
                 random_seed,
                 num_classes=1,
                 ):
        """TODO

        Args:
            TODO

        Returns:
            TODO
        """
        self.meta_batch_size        = meta_batch_size
        self.num_training_samples   = num_training_samples
        self.num_validation_samples = num_validation_samples
        self.num_classes            = num_classes
        self.random_seed            = random_seed

    def set_random_seed(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tf.set_random_seed(self.random_seed)


class Dummy(Dataset):

    name = 'dummy'

    def __init__(self,
                 num_training_samples,
                 num_validation_samples,
                 meta_batch_size,
                 source_dir,
                 input_size,
                 random_seed,
                 num_classes,
                 num_train_batches,
                 num_val_batches,
                 test=False,
                 ):

        self.shape_input = [input_size]
        self.dim_output  = num_classes

        # Create dummy data
        self.training_batch    = tf.ones([meta_batch_size, num_training_samples,   input_size])
        self.validation_batch  = tf.ones([meta_batch_size, num_validation_samples, input_size])
        self.training_labels   = tf.ones([meta_batch_size, num_training_samples,   num_classes])
        self.validation_labels = tf.ones([meta_batch_size, num_validation_samples, num_classes])

    def __call__(self, train=True):
        return self.training_batch, self.training_labels, self.validation_batch, self.validation_labels

    
class Toy(Dataset):

    name = 'toy'

    def __init__(self,
                 num_training_samples,
                 num_validation_samples,
                 meta_batch_size,
                 source_dir,
                 input_size,
                 random_seed,
                 num_total_batches,
                 num_workers,
                 mode,
                 num_classes=1,
                 location=[0., 0.],
                 precision=.1,
                 scale=np.eye(2, dtype=np.float32)/100,
                 dof=4,
                 rejection_threshold=1e-2,
                 test=False,
                 grid_bounds=50.,
                 grid_ticks=100,
                ):
        super().__init__(num_training_samples, num_validation_samples, meta_batch_size, random_seed, num_classes)
        self.mode = mode

        self.dim_output = self.num_classes
        self.shape_input = [2]

        # Normal-Wishart hyperparameters
        self.location  = location
        self.precision = precision
        self.dof       = dof
        self.scale     = scale

        # Rejection threshold for generating negative samples
        self.rejection_threshold = rejection_threshold

        # Parameters to define the activation heatmap grid
        self.grid_bounds = grid_bounds
        self.grid_ticks  = grid_ticks

    @log_function_call("toy data pipeline setup")
    def __call__(self, train=True):
        """TODO

        Args:
            TODO

        Returns:
            TODO
        """
        assert self.num_classes == 1, ">1 classes not yet implemented"

        training_data   = []
        validation_data = []

        training_shape   = [self.num_training_samples]
        validation_shape = [self.num_validation_samples]

        # sample Λ from W(W, ν)
        lambda_dist = tf.contrib.distributions.WishartFull(df=self.dof, scale=self.scale)
        lambda_sample = lambda_dist.sample(name='lambda')
        lambda_inverse = tf.matrix_inverse(lambda_sample)

        # sample µ from N(µ0, (λΛ)^{-1})
        mu_dist = tf.contrib.distributions.MultivariateNormalFullCovariance(self.location, (1/self.precision)*lambda_inverse)
        mu_sample = mu_dist.sample(name='mu')

        # sample an observation x from N(µ, (Λ)^{-1})
        x_dist = tf.contrib.distributions.MultivariateNormalFullCovariance(mu_sample, lambda_inverse)
        training_data = x_dist.sample(sample_shape=training_shape,   name='train_x')
        training_labels_batch = tf.ones([self.meta_batch_size, self.num_training_samples, 1])

        validation_x_sample_pos = x_dist.sample(sample_shape=validation_shape, name='val_x_pos')

        # Generate negatives
        #negative_distribution = tf.contrib.distributions.MultivariateNormalFull(mu_sample, 10*lambda_inverse)
        #stddev = tf.sqrt(tf.diag_part(lambda_inverse))
        s = self.grid_bounds / 2
        low  = [-s, -s] #mu_sample - 8 * stddev
        high = [s,  s] #mu_sample + 8 * stddev
        negative_distribution = tf.contrib.distributions.Uniform(low, high)

        validation_x_sample_neg_list = []
        for _ in range(validation_shape[0]):
            i = negative_distribution.sample()
            c = lambda x: tf.cast(False, tf.bool) #lambda x: tf.reduce_any(tf.greater(x_dist.prob(x), self.rejection_threshold))  # TODO: no longer exits while loop?
            b = lambda x: negative_distribution.sample()
            validation_x_sample_neg_list += [tf.while_loop(c, b, [i])]

        validation_x_sample_neg = tf.stack(validation_x_sample_neg_list)

        validation_data = tf.concat([validation_x_sample_pos, validation_x_sample_neg], axis=0)

        validation_pos_labels_batch = tf.ones([self.meta_batch_size,  self.num_validation_samples, 1])
        validation_neg_labels_batch = tf.zeros([self.meta_batch_size, self.num_validation_samples, 1])
        validation_labels_batch     = tf.concat([validation_pos_labels_batch, validation_neg_labels_batch], axis=1)

        training_batch, validation_batch = tf.train.batch([training_data, validation_data],
                                                          batch_size=self.meta_batch_size,
                                                          num_threads=1,
                                                          capacity=256 + 3 * self.meta_batch_size,
                                                         )

        if self.mode in ['val', 'test']:

            # Also add grid input in order to investigate network activation
            s = self.grid_bounds
            n = self.grid_ticks

            data_batch = tf.stack(tf.meshgrid(
                tf.linspace(-s, s, n),
                tf.linspace(-s, s, n),
            ), -1)
            data_batch = tf.expand_dims(tf.reshape(data_batch, [-1, 2]), axis=0)
            labels_batch = tf.zeros([self.meta_batch_size, n**2, 1])

            validation_batch        = tf.concat([validation_batch,        data_batch],   axis=1)
            validation_labels_batch = tf.concat([validation_labels_batch, labels_batch], axis=1)

        return {
            'training data':     training_batch, 
            'training labels':   training_labels_batch, 
            'validation data':   validation_batch, 
            'validation labels': validation_labels_batch,
        }
