from itertools import repeat
import logging
from multiprocessing import Pool
from nltk.corpus import wordnet as wn
import numpy as np
import os
import pickle
import random

from cogsci_2019_concept_learning.data.dataset import Dataset
from cogsci_2019_concept_learning.utils.utils_gen import br, flatten, log_function_call, multi_pop, ss
from cogsci_2019_concept_learning.utils.utils_np import shuffle_in_unison, dense_to_one_hot


class Classification(Dataset):

    def __init__(self,
                 num_training_samples,
                 num_validation_samples,
                 meta_batch_size,
                 source_dir,
                 output_dir,
                 input_size,
                 num_classes,
                 random_seed,
                 num_total_batches,
                 mode,
                 num_workers,
                 use_reference_batch,
                 size_reference_batch=10,
                 ):
        """A subset of classes and images from ImageNet.

        Args:
            TODO

        Returns:
            TODO
        """
        super().__init__(num_training_samples, num_validation_samples, meta_batch_size, random_seed, num_classes)

        self.shape_input = [input_size, input_size, 3]
        self.dim_input = np.prod(self.shape_input)
        self.dim_output = self.num_classes

        self.num_total_batches = num_total_batches
        self.num_workers = num_workers
        self.source_dir = source_dir
        self.output_dir = output_dir

        self.use_reference_batch = use_reference_batch
        self.size_reference_batch = size_reference_batch

        if mode == 'train':
            label_to_images_map = self.train_label_to_image_map
            self.label_to_positive_leaf_labels_map = self.train_labels_to_leaf_labels_map
        elif mode == 'val':
            label_to_images_map = self.val_label_to_image_map
            self.label_to_positive_leaf_labels_map = self.val_labels_to_leaf_labels_map
        elif mode == 'test':
            label_to_images_map = self.test_label_to_image_map
            self.label_to_positive_leaf_labels_map = self.test_labels_to_leaf_labels_map
        self.label_to_negative_leaf_labels_map = self.label_to_negative_examples_map
        self.mode = mode

        # Preprocessing speed-ups:
        # More efficient to index a numpy array
        # Join the source dir ahead of time
        self.label_to_dir_map = {label: os.path.join(source_dir, label) for label in os.listdir(source_dir)}
        self.label_to_images_map = {label: np.array([os.path.join(self.label_to_dir_map[label], image)
                                                     for image in label_to_images_map[label]])
                                    for label in label_to_images_map}

    @property
    def train_label_to_image_map(self):
        raise NotImplementedError("Abstract method.")

    @property
    def val_label_to_image_map(self):
        raise NotImplementedError("Abstract method.")

    @property
    def test_label_to_image_map(self):
        raise NotImplementedError("Abstract method.")

    def summarize_data(self, data_tensor, name):
        input_num_dims = len(self.shape_input)
        begin = [0, 0] + [0] * input_num_dims
        size = [1, -1] + [-1] * input_num_dims
        max_outputs = int(data_tensor.get_shape().as_list()[2])
        tf.summary.image(name, tf.squeeze(tf.slice(data_tensor, begin, size), squeeze_dims=[0]),
                         max_outputs=max_outputs)

    def get_reference_batch(self):

        from scipy.ndimage import imread

        # Sample images randomly from the training dataset
        choice_labels = np.random.choice(list(self.train_label_to_image_map.keys()), 10)
        choice_images = list(flatten([np.random.choice(self.train_label_to_image_map[label]) for label in choice_labels]))

        images_file = [os.path.join(self.label_to_dir_map[choice_labels[i]], choice_images[i]) for i in range(len(choice_images))]
        images_string = [tf.read_file(i) for i in images_file]
        images_decoded = [tf.image.decode_image(i, channels=self.shape_input[-1]) for i in images_string]
        [i.set_shape(self.shape_input) for i in images_decoded]
        #images_resized = [tf.image.resize_images(i, self.shape_input[:2]) for i in images_decoded]
        images_resized = images_decoded

        reference_batch = tf.stack(images_resized)
        reference_batch = tf.cast(reference_batch, tf.float32) / 255.0

        return reference_batch

    def create_queue(self, filename_queue, batch_shape, name):
        """Create a queue from the list of files written to filename."""

        # DATA QUEUES
        num_preprocess_threads = 1  # used to be 8!!! (which leads to incorrect behavior)
        min_after_dequeue = 10000
        batch_capacity_multiplier = 20

        examples_per_batch = np.prod(batch_shape)
        queue_capacity = min_after_dequeue + batch_capacity_multiplier

        reader = tf.TextLineReader()
        _, row = reader.read(filename_queue)
        filename_queue = tf.train.string_input_producer([row], shuffle=False)

        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)

        image = tf.image.decode_jpeg(image_file, channels=self.shape_input[-1])
        image.set_shape(self.shape_input)
        image = tf.cast(image, tf.float32) / 255.0

        images_batch = tf.train.batch([image],
                                      batch_size=self.meta_batch_size * examples_per_batch,
                                      num_threads=num_preprocess_threads,
                                      capacity=queue_capacity,
                                      name=name,
                                      )
        images_batch = tf.reshape(images_batch, [self.meta_batch_size] + batch_shape + self.shape_input)

        return images_batch

    def get_images(self, leaves, nb_samples):

        num_per_leaf = nb_samples // len(leaves)
        if nb_samples % len(leaves) > 0:
            num_per_leaf += 1

        images = []
        for leaf in leaves:
            choices = self.label_to_images_map[leaf]
            choice_idcs = np.random.randint(len(choices), size=num_per_leaf)
            images += list(choices[choice_idcs])

        if len(images) > nb_samples:
            np.random.shuffle(images)
            return images[:nb_samples]

        else:
            return images

class BaselineClassification(Classification):

    def __init__(self,
                 num_training_samples,
                 num_validation_samples,
                 meta_batch_size,
                 source_dir,
                 output_dir,
                 input_size,
                 num_classes,
                 random_seed,
                 num_total_batches,
                 mode,
                 num_workers,
                 use_reference_batch,
                 size_reference_batch=10,
                 ):
        """TODO

        Args:
            TODO
            pos_and_neg: include both positive and negative examples in fast update batch

        Returns:
            TODO
        """
        super().__init__(num_training_samples, num_validation_samples, meta_batch_size, source_dir, output_dir, input_size, num_classes, random_seed, num_total_batches, mode, num_workers, use_reference_batch, size_reference_batch)
        self.dim_output = self.num_classes

    @log_function_call("baseline multiway classification task data pipeline setup")
    def __call__(self):
        """TODO

        Args:
            TODO

        Returns:
            TODO
        """
        self.set_random_seed()

        # Baseline trains on multi-way classification
        leaves = list(self.label_to_images_map.keys())
        num_labels = len(leaves)
        num_samples = self.num_training_samples

        # Construct image and label lists
        images = []
        labels = []
        for i, leaf in enumerate(leaves):
            images += [os.path.join(self.label_to_dir_map[leaf], image) for image in self.label_to_images_map[leaf]]
            labels += [i] * len(self.label_to_images_map[leaf])

        # Convert to TF Tensors
        all_filenames_tensor = tf.reshape(tf.convert_to_tensor(images), [-1])
        all_labels_tensor    = tf.reshape(tf.convert_to_tensor(labels), [-1])
        one_hot_labels_tensor = tf.one_hot(all_labels_tensor, num_labels)

        label = tf.train.input_producer(one_hot_labels_tensor, shuffle=False).dequeue()
        label.set_shape([num_labels])

        filename = tf.train.string_input_producer(all_filenames_tensor, shuffle=False)
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename)

        image = tf.image.decode_jpeg(image_file, channels=self.shape_input[-1])
        image.set_shape(self.shape_input)
        image = tf.cast(image, tf.float32) / 255.0

        # DATA QUEUE
        num_preprocess_threads = 1
        min_after_dequeue = 10000
        batch_capacity_multiplier = 20
        examples_per_batch = num_samples
        queue_capacity = min_after_dequeue + batch_capacity_multiplier

        images_batch, labels_batch = tf.train.shuffle_batch([image, label],
                batch_size=self.meta_batch_size * examples_per_batch,
                num_threads=num_preprocess_threads,
                capacity=queue_capacity,
                name='data',
                min_after_dequeue=min_after_dequeue
                )

        # Reference batch for virtual batch norm
        if self.use_reference_batch is True:
            images_batch = tf.concat([images_batch, self.get_reference_batch()], axis=0)
            images_shape = examples_per_batch + self.size_reference_batch
        else:
            images_shape = examples_per_batch

        images_batch = tf.reshape(images_batch, [self.meta_batch_size, images_shape] + self.shape_input)
        labels_batch = tf.reshape(labels_batch, [self.meta_batch_size, examples_per_batch] + [num_labels])

        return {
            'training data': None,
            'training labels': None,
            'validation data': images_batch,
            'validation labels': labels_batch,
        }


class BinaryClassification(Classification):


    @property
    def neg_pos_ratio(self):
        raise NotImplementedError

    @log_function_call("binary classification task data pipeline setup")
    def __call__(self, node_to_positives_map, node_to_negatives_map):

        assert num_classes == 1 or num_classes == 2
        self.set_random_seed()

        #node_to_positives_map = self.label_to_positive_leaf_labels_map
        #node_to_negatives_map = self.label_to_negative_leaf_labels_map

        # Batchsize params
        num_training_samples_pos = self.num_training_samples
        num_training_samples_neg = int(self.num_training_samples * self.neg_pos_ratio)
        num_validation_samples   = self.num_validation_samples

        training_batch_size = num_training_samples_pos + num_training_samples_neg
        validation_batch_size = num_validation_samples * 2

        # Sample nodes to define training and validation trials
        sampled_nodes = list(np.random.choice(list(node_to_positives_map.keys()), size=self.num_total_batches, replace=True))

        sampled_positive_leaf_lists = [node_to_positives_map[sampled_node] for sampled_node in sampled_nodes]
        sampled_negative_leaf_lists = [node_to_negatives_map[sampled_node] for sampled_node in sampled_nodes]

        with Pool(self.num_workers) as pool:

            positive_images = pool.starmap_async(self.get_images, zip(sampled_positive_leaf_lists, repeat(num_training_samples_pos + num_validation_samples))).get()
            negative_images = pool.starmap_async(self.get_images, zip(sampled_negative_leaf_lists, repeat(num_training_samples_neg + num_validation_samples))).get()

        assert len(positive_images) == len(negative_images)

        training_images, validation_images = [], []
        for i in range(len(positive_images)):

            pos = positive_images[i]
            neg = negative_images[i]

            assert len(pos) == num_training_samples_pos + num_validation_samples
            assert len(neg) == num_training_samples_neg + num_validation_samples

            training_images   += [pos[:num_training_samples_pos] + neg[:num_training_samples_neg]]
            validation_images += [pos[num_training_samples_pos:] + neg[num_training_samples_neg:]]

            assert len(training_images[i]) == training_batch_size
            assert len(validation_images[i]) == validation_batch_size

        # Write the list of filenames to file
        training_images   = flatten(training_images)
        validation_images = flatten(validation_images)

        return training_images, validation_images

        """
        training_images_filename = os.path.join(self.output_dir, '%s_training.csv' % self.mode)
        validation_images_filename = os.path.join(self.output_dir, '%s_validation.csv' % self.mode)
        with open(training_images_filename, 'w') as f:
            f.write('\n'.join(training_images))
        with open(validation_images_filename, 'w') as f:
            f.write('\n'.join(validation_images))

        training_images_filename_queue   = tf.train.string_input_producer([training_images_filename])
        validation_images_filename_queue = tf.train.string_input_producer([validation_images_filename])

        training_images_batch   = self.create_queue(training_images_filename_queue,   [training_batch_size],   'training_batch')
        validation_images_batch = self.create_queue(validation_images_filename_queue, [validation_batch_size], 'validation_batch')

        # Reference batch for virtual batch norm
        if self.use_reference_batch is True:
            tiled_reference_batch = tf.tile(tf.expand_dims(self.get_reference_batch(), 0), [self.meta_batch_size, 1] +  [1] * len(self.shape_input))
            training_images_batch   = tf.concat([training_images_batch,   tiled_reference_batch], axis=1)
            validation_images_batch = tf.concat([validation_images_batch, tiled_reference_batch], axis=1)

        # LABELS
        training_positive_labels_batch = tf.ones([self.meta_batch_size,  num_training_samples_pos, 1])
        training_negative_labels_batch = tf.zeros([self.meta_batch_size, num_training_samples_neg, 1])
        training_labels_batch = tf.concat([training_positive_labels_batch, training_negative_labels_batch], axis=1)

        validation_positive_labels_batch = tf.ones([self.meta_batch_size,  num_validation_samples, 1])
        validation_negative_labels_batch = tf.zeros([self.meta_batch_size, num_validation_samples, 1])
        validation_labels_batch = tf.concat([validation_positive_labels_batch, validation_negative_labels_batch], axis=1)

        # Check in the case of virtual batch norm
        #assert_matching_dims(training_images_batch,   training_labels_batch,   dims=[0, 1])
        #assert_matching_dims(validation_images_batch, validation_labels_batch, dims=[0, 1])

        #self.summarize_data(training_images_batch,   'training_images')
        #self.summarize_data(validation_images_batch, 'validation_images')

        return {
            'training data': training_images_batch,
            'training labels': training_labels_batch,
            'validation data': validation_images_batch,
            'validation labels': validation_labels_batch,
        }
        """


class OneToOneBinaryClassification(BinaryClassification):
    @property
    def neg_pos_ratio(self):
        return 1.0


class FiveToFourBinaryClassification(BinaryClassification):
    @property
    def neg_pos_ratio(self):
        return 0.8


class FiveToThreeBinaryClassification(BinaryClassification):
    @property
    def neg_pos_ratio(self):
        return 0.6


class FiveToTwoBinaryClassification(BinaryClassification):
    @property
    def neg_pos_ratio(self):
        return 0.4


class FiveToOneBinaryClassification(BinaryClassification):
    @property
    def neg_pos_ratio(self):
        return 0.2


class MultiWayClassification(Classification):

    def __init__(self,
                 num_training_samples,
                 num_validation_samples,
                 meta_batch_size,
                 source_dir,
                 output_dir,
                 input_size,
                 num_classes,
                 random_seed,
                 num_total_batches,
                 mode,
                 num_workers,
                 use_reference_batch,
                 size_reference_batch=10,
                 ):
        """TODO

        Args:
            TODO

        Returns:
            TODO
        """
        assert num_classes > 1
        super().__init__(num_training_samples, num_validation_samples, meta_batch_size, source_dir, output_dir, input_size, num_classes, random_seed, num_total_batches, mode, num_workers, use_reference_batch, size_reference_batch)

    @log_function_call("multi-class classification task data pipeline setup")
    def __call__(self):
        """TODO"""
        self.set_random_seed()

        node_to_positives_map = self.label_to_positive_leaf_labels_map

        # Sample nodes to define training and validation trials
        sampled_nodes = [list(np.random.choice(list(node_to_positives_map.keys()), size=self.num_classes, replace=False)) for _ in range(self.num_total_batches)]
        sampled_leaf_lists = [node_to_positives_map[node] for batch in sampled_nodes for node in batch]

        with Pool(self.num_workers) as pool:
            examples = pool.starmap_async(self.get_images, zip(sampled_leaf_lists, repeat(self.num_training_samples + self.num_validation_samples))).get()

        training_images, validation_images = [], []
        for batch in examples:
            assert len(batch) == self.num_training_samples + self.num_validation_samples
            validation_images += [batch[self.num_training_samples:]]

        # Write the list of filenames to file
        training_images   = flatten(training_images)
        validation_images = flatten(validation_images)

        return training_images, validation_images

        """
        training_images_filename = os.path.join(self.output_dir, '%s_training.csv' % self.mode)
        validation_images_filename = os.path.join(self.output_dir, '%s_validation.csv' % self.mode)
        with open(training_images_filename, 'w') as f:
            f.write('\n'.join(training_images))
        with open(validation_images_filename, 'w') as f:
            f.write('\n'.join(validation_images))

        training_images_filename_queue   = tf.train.string_input_producer([training_images_filename])
        validation_images_filename_queue = tf.train.string_input_producer([validation_images_filename])

        training_images_batch   = self.create_queue(training_images_filename_queue,   [self.num_training_samples * self.num_classes],   'training_batch')
        validation_images_batch = self.create_queue(validation_images_filename_queue, [self.num_validation_samples * self.num_classes], 'validation_batch')

        # Reference batch for virtual batch norm
        if self.use_reference_batch is True:
            tiled_reference_batch = tf.tile(tf.expand_dims(self.get_reference_batch(), 0), [self.meta_batch_size, 1] +  [1] * len(self.shape_input))
            training_images_batch   = tf.concat([training_images_batch,   tiled_reference_batch], axis=1)
            validation_images_batch = tf.concat([validation_images_batch, tiled_reference_batch], axis=1)

        # LABELS
        training_labels_batch   = tf.one_hot(tf.reshape(tf.tile(tf.expand_dims(tf.expand_dims(tf.range(self.num_classes), -1), 0), [self.meta_batch_size, 1, self.num_training_samples]),   [self.meta_batch_size, -1]), depth=self.num_classes)
        validation_labels_batch = tf.one_hot(tf.reshape(tf.tile(tf.expand_dims(tf.expand_dims(tf.range(self.num_classes), -1), 0), [self.meta_batch_size, 1, self.num_validation_samples]), [self.meta_batch_size, -1]), depth=self.num_classes)

        # Check in the case of virtual batch norm
        #assert_matching_dims(training_images_batch,   training_labels_batch,   dims=[0, 1])
        #assert_matching_dims(validation_images_batch, validation_labels_batch, dims=[0, 1])

        #self.summarize_data(training_images_batch,   'training_images')
        #self.summarize_data(validation_images_batch, 'validation_images')

        return {
            'training data': training_images_batch,
            'training labels': training_labels_batch,
            'validation data': validation_images_batch,
            'validation labels': validation_labels_batch,
        }
        """


class UnaryClassification(Classification):

    def __init__(self,
                 num_training_samples,
                 num_validation_samples,
                 meta_batch_size,
                 source_dir,
                 output_dir,
                 input_size,
                 num_classes,
                 random_seed,
                 num_total_batches,
                 mode,
                 num_workers,
                 use_reference_batch,
                 size_reference_batch=10,
                 ):
        """TODO

        Args:
            TODO
            pos_and_neg: include both positive and negative examples in fast update batch

        Returns:
            TODO
        """
        assert num_classes == 1 or num_classes == 2
        super().__init__(num_training_samples, num_validation_samples, meta_batch_size, source_dir, output_dir, input_size, num_classes, random_seed, num_total_batches, mode, num_workers, use_reference_batch, size_reference_batch)

    @log_function_call("unary classification task data pipeline setup")
    def __call__(self):
        """TODO

        Args:
            TODO

        Returns:
            TODO
        """
        self.set_random_seed()

        node_to_positives_map = self.label_to_positive_leaf_labels_map
        node_to_negatives_map = self.label_to_negative_leaf_labels_map

        # Batchsize params
        num_training_samples   = self.num_training_samples
        num_validation_samples = self.num_validation_samples

        # Sample nodes to define training and validation trials
        sampled_nodes = list(np.random.choice(list(node_to_positives_map.keys()), size=self.num_total_batches, replace=True))

        sampled_positive_leaf_lists = [node_to_positives_map[sampled_node] for sampled_node in sampled_nodes]
        sampled_negative_leaf_lists = [node_to_negatives_map[sampled_node] for sampled_node in sampled_nodes]

        with Pool(self.num_workers) as pool:

            positive_images = pool.starmap_async(self.get_images, zip(sampled_positive_leaf_lists, repeat(num_training_samples + num_validation_samples))).get()
            negative_images = pool.starmap_async(self.get_images, zip(sampled_negative_leaf_lists, repeat(num_validation_samples))).get()

        assert len(positive_images) == len(negative_images)

        training_images, validation_images = [], []
        for i in range(len(positive_images)):

            pos = positive_images[i]
            neg = negative_images[i]

            assert len(pos) == num_training_samples + num_validation_samples
            assert len(neg) == num_validation_samples

            training_images   += [pos[:num_training_samples]]
            validation_images += [pos[num_training_samples:] + neg]

            assert len(training_images[i]) == num_training_samples
            assert len(validation_images[i]) == num_validation_samples * 2

        training_images   = flatten(training_images)
        validation_images = flatten(validation_images)

        return training_images, validation_images

        """
        training_images_filename = os.path.join(self.output_dir, '%s_training.csv' % self.mode)
        validation_images_filename = os.path.join(self.output_dir, '%s_validation.csv' % self.mode)
        with open(training_images_filename, 'w') as f:
            f.write('\n'.join(training_images))
        with open(validation_images_filename, 'w') as f:
            f.write('\n'.join(validation_images))

        training_images_filename_queue   = tf.train.string_input_producer([training_images_filename])
        validation_images_filename_queue = tf.train.string_input_producer([validation_images_filename])

        training_images_batch   = self.create_queue(training_images_filename_queue,   [self.num_training_samples],       'training_batch')
        validation_images_batch = self.create_queue(validation_images_filename_queue, [self.num_validation_samples * 2], 'validation_batch')

        # Reference batch for virtual batch norm
        if self.use_reference_batch is True:
            tiled_reference_batch = tf.tile(tf.expand_dims(self.get_reference_batch(), 0), [self.meta_batch_size, 1] +  [1] * len(self.shape_input))
            training_images_batch   = tf.concat([training_images_batch,   tiled_reference_batch], axis=1)
            validation_images_batch = tf.concat([validation_images_batch, tiled_reference_batch], axis=1)

        # LABELS
        training_labels_batch = tf.ones([self.meta_batch_size, self.num_training_samples, 1])

        validation_positive_labels_batch = tf.ones([self.meta_batch_size,  self.num_validation_samples, 1])
        validation_negative_labels_batch = tf.zeros([self.meta_batch_size, self.num_validation_samples, 1])
        validation_labels_batch = tf.concat([validation_positive_labels_batch, validation_negative_labels_batch], axis=1)

        # Check in the case of virtual batch norm
        #assert_matching_dims(training_images_batch,   training_labels_batch,   dims=[0, 1])
        #assert_matching_dims(validation_images_batch, validation_labels_batch, dims=[0, 1])

        #self.summarize_data(training_images_batch,   'training_images')
        #self.summarize_data(validation_images_batch, 'validation_images')

        return {
            'training data': training_images_batch,
            'training labels': training_labels_batch,
            'validation data': validation_images_batch,
            'validation labels': validation_labels_batch,
        }
        """
