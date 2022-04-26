from glob import glob
import json
import logging
import numpy as np
import os
import pickle
from PIL import Image
import random
import tensorflow as tf

from cogsci_2019_concept_learning.utils.utils_gen import log_function_call


HUMAN_COMP_DIR = os.path.join(os.path.dirname(__file__))
IMAGES_DIR     = os.path.join(HUMAN_COMP_DIR, 'imagenet')


def load_data_file(filename, input_size):
    im = Image.open(os.path.join(IMAGES_DIR, filename))
    #im = im.resize((input_size, input_size), resample=Image.LANCZOS)
    im = np.array(im)
    if len(im.shape) < 3:
        im = np.stack([im, im, im], axis=2)
    return np.array(im, dtype=np.float32) / 255.


def get_match_types(matches):
    ms = ['sub'] * 2 + ['basic-level'] * 4 + ['super'] * 16
    assert len(ms) == 24
    return ms


def process_trial(trial_data, gt_data, input_size):

    test_images = trial_data['test_images']
    train_images = trial_data['train_images']
    trial_type = trial_data['trial_type2']
    condition = trial_data['condition']
    responses = trial_data['responses']
    response_gt_index = [gt_data['test images'].index(x) for x in responses]
    test_gt_index     = [gt_data['test images'].index(x) for x in test_images]

    trial_type = {
        '1_ex':  1,
        '3_ex':  3,
        '5_ex':  5,
        '10_ex': 10,
    }[trial_type]

    match_type_to_label = {
        'subordinate_match': 0,
        'basic_match': 1,
        'superordinate_match': 2,
        'out_of_concept': 3,
    }

    gt_labels = []
    response_labels = []
    for idx in test_gt_index:
        if idx in [0, 1]:
            match_type = 'subordinate_match'
        elif idx in [2, 3]:
            match_type = 'basic_match'
        elif idx in [4, 5, 6, 7]:
            match_type = 'superordinate_match'
        elif idx in range(8, 24):
            match_type = 'out_of_concept'
        else:
            raise Exception
        gt_labels += [match_type_to_label[match_type]]

        if idx in response_gt_index:
            response_labels += [1]
        else:
            response_labels += [0]

    assert len(gt_labels) == len(response_labels)
    gt_labels = np.array(gt_labels)
    response_labels = np.array(response_labels)

    support = np.array([load_data_file(im, input_size) for im in train_images])
    test    = np.array([load_data_file(im, input_size) for im in test_images])

    return trial_type, condition, support, test, response_labels, gt_labels


def process_data_row(data_row, spec, input_size):

    trial_ID = data_row['data'][0]['trial_ID']
    gt_data = spec[trial_ID]

    results = {}

    results[1] = {}
    results[3] = {}
    results[5] = {}
    results[10] = {}

    for num_examps in results.keys():
        for condition in ['subordinate_condition', 'basic-level_condition', 'superordinate_condition']:
            results[num_examps][condition] = {}
            results[num_examps][condition]['train_set'] = []
            results[num_examps][condition]['test_set'] = []
            results[num_examps][condition]['response_labels'] = []
            results[num_examps][condition]['gt_labels'] = []

    num_trials = 8
    for i in range(num_trials):
        trial_type, condition, support, test, response_labels, gt_labels = process_trial(data_row['data'][i], gt_data[i], input_size)

        results[trial_type][condition]['train_set'].append(support)
        results[trial_type][condition]['test_set'].append(test)
        results[trial_type][condition]['response_labels'].append(response_labels)
        results[trial_type][condition]['gt_labels'].append(gt_labels)

    return results


class HumanCompDataset(object):

    def __init__(self,
                 input_size,
                 random_seed,
                 ):
        """A human comparison dataset.

        Args:
            TODO

        Returns:
            TODO
        """
        self.shape_input = [input_size, input_size, 3]
        self.dim_input = np.prod(self.shape_input)

        # Set up the data
        results = []
        amturk_dir = os.path.join(HUMAN_COMP_DIR, 'amturk_results')
        for exp_dir in os.listdir(amturk_dir):
            results_pkl = glob(os.path.join(amturk_dir, exp_dir, '*.json'))
            spec_pkl    = glob(os.path.join(amturk_dir, exp_dir, '*.pkl'))

            assert len(results_pkl) == 1, "Too many files in directory"
            assert    len(spec_pkl) == 1, "Too many files in directory"

            results_pkl = results_pkl[0]
            spec_pkl    = spec_pkl[0]

            # Open the experimental specifications file
            with open(spec_pkl, 'rb') as f:
                spec = pickle.load(f)

            # Process the dataset
            for line in open(results_pkl, 'r'):
                data_row = json.loads(line)
                if data_row['data']:
                    results += [process_data_row(data_row, spec, input_size)]

        # Collapse results into a single dictionary
        self.results = {}

        for num_examps in self.num_training_samples:
            self.results[num_examps] = {}

        for num_examps in self.results.keys():
            for condition in self.conditions:
                self.results[num_examps][condition] = {}
                self.results[num_examps][condition]['train_set']  = []
                self.results[num_examps][condition]['test_set']  = []
                self.results[num_examps][condition]['response_labels'] = []
                self.results[num_examps][condition]['gt_labels'] = []

        for result in results:
            for num_examps in result.keys():
                for condition in result[num_examps]:
                    for dataset in result[num_examps][condition]:
                        self.results[num_examps][condition][dataset] += result[num_examps][condition][dataset]

    @property
    def match_types(self):
        return ['sub. match', 'bas. match', 'sup. match', 'ood. match']

    @property
    def num_training_samples(self):
        return [1, 3, 5, 10]

    @property
    def conditions(self):
        return ['subordinate_condition', 'basic-level_condition',   'superordinate_condition']

    @log_function_call("human data comparison pipeline setup")
    def __call__(self, num_training_samples, condition):
        """TODO

        Args:
            TODO

        Returns:
            TODO
        """
        assert num_training_samples in self.num_training_samples, '%d not in permissible condition types' % num_training_samples
        assert condition in self.conditions, '%s not in permissible condition types' % condition

        num_batches = len(self.results[num_training_samples][condition]['train_set'])
        assert num_batches == len(self.results[num_training_samples][condition]['test_set'])
        assert num_batches == len(self.results[num_training_samples][condition]['response_labels'])
        assert num_batches == len(self.results[num_training_samples][condition]['gt_labels'])

        all_batches = []
        for i in range(num_batches):
            all_batches.append({
                'training data':   self.results[num_training_samples][condition]['train_set'][i][np.newaxis, :],
                'training labels': np.ones((1, num_training_samples, 1)),
                'validation data': self.results[num_training_samples][condition]['test_set'][i][np.newaxis, :],
                'response labels': self.results[num_training_samples][condition]['response_labels'][i][np.newaxis, :, np.newaxis],
                'gt labels':       self.results[num_training_samples][condition]['gt_labels'][i][np.newaxis, :, np.newaxis],
            })

        return all_batches
