import argparse
import inspect
import logging
from multiprocessing import cpu_count
import numpy as np
import os
import random
import tensorflow as tf

from cogsci_2019_concept_learning.core.acts import lrelu, prelu, relu, sigmoid, tanh
from cogsci_2019_concept_learning.core.norm import batch_norm, batch_renorm, identity_norm, layer_norm, virtual_batch_norm
from cogsci_2019_concept_learning.core.opt import adam_9_opt, adam_99_opt
from cogsci_2019_concept_learning.core.update_rules import adam, rmsprop, sgd


def create_exp_id(args):

    exp_id = ('%s.'              % args.prefix) if args.prefix else ''

    exp_id += '%s'               % args.model_arch.name
    exp_id += '.nc_%d'           % args.num_classes

    exp_id += '.%s'              % args.norm.name
    exp_id += '.init_%s'         % type(args.init).__name__
    exp_id += '.nonlin_%s'       % args.nonlin.__name__

    exp_id += '.mbs_%d'          % args.meta_batch_size
    exp_id += ('.mcn_%.2f'       % args.meta_clip_norm) if args.meta_clip_norm else ''
    exp_id += '.mlr_%.5f'        % args.meta_step_size
    exp_id += '.mop_%s'          % args.meta_optimizer.name

    exp_id += '.uni_%d'          % args.num_updates
    exp_id += '.utbs_%d'         % args.update_training_batch_size
    exp_id += '.uvbs_%d'         % args.update_validation_batch_size
    exp_id += ('.ucn_%.2f'       % args.update_clip_norm) if args.update_clip_norm else ''
    exp_id += ('.uss_%.5f'       % args.update_step_size) if args.update_step_size else '_uss_adapt'
    exp_id += '.uop_%s'          % args.update_optimizer.name

    return exp_id


class ParseType(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.types[values.lower()])


class ParseNorm(ParseType):

    types = {
        'batch_norm':         batch_norm,
        'batch_renorm':       batch_renorm,
        'layer_norm':         layer_norm,
        'virtual_batch_norm': virtual_batch_norm,
        'none':               identity_norm,
    }


class ParseDataset(ParseType):

    types = {}

    import cogsci_2019_concept_learning.data.dataset as dataset
    types.update({x[1].name: x[1] for x in inspect.getmembers(dataset) if inspect.isclass(x[1]) and hasattr(x[1], 'name')})

    import cogsci_2019_concept_learning.data.imagenet as imagenet
    types.update({x[1].name: x[1] for x in inspect.getmembers(imagenet) if inspect.isclass(x[1]) and hasattr(x[1], 'name')})

    import cogsci_2019_concept_learning.data.human as human
    types.update({x[1].name: x[1] for x in inspect.getmembers(human) if inspect.isclass(x[1]) and hasattr(x[1], 'name')})


class ParseNonLin(ParseType):

    types = {
        'relu':    relu,
        'lrelu':   lrelu,
        'prelu':   prelu,
        'sigmoid': sigmoid,
        'tanh':    tanh,
    }


class ParseInit(ParseType):

    types = {
        'trunc_norm': tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
        'xavier':     tf.contrib.layers.xavier_initializer(),
        'orthogonal': tf.orthogonal_initializer(),
    }


class ParseModelArch(ParseType):

    types = {}

    #import cogsci_2019_concept_learning.core.baseline as baseline
    #types.update({x[1].name: x[1] for x in inspect.getmembers(baseline) if inspect.isclass(x[1]) and hasattr(x[1], 'name')})

    import cogsci_2019_concept_learning.core.maml as maml
    types.update({x[1].name: x[1] for x in inspect.getmembers(maml) if inspect.isclass(x[1]) and hasattr(x[1], 'name')})

    import cogsci_2019_concept_learning.core.metric_learner as metric_learner
    types.update({x[1].name: x[1] for x in inspect.getmembers(metric_learner) if inspect.isclass(x[1]) and hasattr(x[1], 'name')})


class ParseUpdateOpt(ParseType):

    types = {
        'adam':    adam,
        'rmsprop': rmsprop,
        'sgd':     sgd,
    }


class ParseMetaOpt(ParseType):

    types = {
        'adam_9':  adam_9_opt,
        'adam_99': adam_99_opt,
    }


def parse_args(args):

    parser = argparse.ArgumentParser(description='Process command-line arguments.')
    parser.add_argument('--random_seed', default=12)
    parser.add_argument('--prefix',      default='')

    ######################### DATASET #########################################
    data_group = parser.add_argument_group('data_params', 'data hyperparameters')
    data_group.add_argument('--data_source_dir', default=None)
    data_group.add_argument('--input_size',      required=True,       type=int)
    data_group.add_argument('--num_workers',     default=cpu_count(), type=int)
    data_group.add_argument('--train_dataset',   required=True, action=ParseDataset, choices=ParseDataset.types.keys())
    data_group.add_argument('--val_dataset',     required=True, action=ParseDataset, choices=ParseDataset.types.keys())
    data_group.add_argument('--test_dataset',    default=None,  action=ParseDataset, choices=ParseDataset.types.keys())

    ######################### MODEL ARCHITECTURE ##############################
    model_group = parser.add_argument_group('model_params', 'model hyperparameters')
    model_group.add_argument('--num_classes', type=int,              default=5)
    model_group.add_argument('--init',        action=ParseInit,      choices=ParseInit.types.keys(),      default=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    model_group.add_argument('--norm',        action=ParseNorm,      choices=ParseNorm.types.keys(),      default=identity_norm)
    model_group.add_argument('--nonlin',      action=ParseNonLin,    choices=ParseNonLin.types.keys(),    default=lrelu)
    model_group.add_argument('--model_arch',  action=ParseModelArch, choices=ParseModelArch.types.keys(), required=True)


    ######################### TRAINING ########################################
    train_group = parser.add_argument_group('train_params', 'training hyperparameters')
    train_group.add_argument('--train',               action='store_true')
    train_group.add_argument('--test',                action='store_true')
    train_group.add_argument('--human_comp',          action='store_true')
    train_group.add_argument('--resume',              action='store_true')
    train_group.add_argument('--num_train_iters',     type=int, default=2000)
    train_group.add_argument('--num_train_batches',   type=int, default=100000)
    train_group.add_argument('--num_val_batches',     type=int, default=10000)
    train_group.add_argument('--num_test_batches',    type=int, default=10000)

    meta_group = train_group.add_argument_group('meta_params', 'Hyperparameters for the meta-level optimization.')
    meta_group.add_argument('--meta_optimizer',  action=ParseMetaOpt, default=adam_9_opt)
    meta_group.add_argument('--meta_step_size',  type=float,          default=0.001)
    meta_group.add_argument('--meta_batch_size', type=int,            default=1)
    meta_group.add_argument('--meta_clip_norm',  type=float,          default=None)

    update_group = train_group.add_argument_group('update_params', 'Hyperparameters for the gradient update.')
    update_group.add_argument('--update_optimizer',             action=ParseUpdateOpt, default=sgd)
    update_group.add_argument('--num_updates',                  type=int,              default=1)
    update_group.add_argument('--update_validation_batch_size', type=int,              default=16)
    update_group.add_argument('--update_training_batch_size',   type=int,              default=16)
    update_group.add_argument('--update_clip_norm',             type=float,            default=None)
    update_group.add_argument('--update_step_size',             type=float,            default=None,
                              help='Leave unspecified to learn the gradient update step size during training; if specified, the fixed size of the gradient step.')

    ######################### LOGGING #########################################
    log_group = parser.add_argument_group('log_params', 'logging hyperparameters')
    log_group.add_argument('--summary_dir',         type=str, default='logs')
    log_group.add_argument('--save_dir',            type=str, default='checkpoints')
    log_group.add_argument('--summary_interval',    type=int, default=None)
    log_group.add_argument('--save_interval',       type=int, default=None)
    log_group.add_argument('--print_interval',      type=int, default=None)
    log_group.add_argument('--test_print_interval', type=int, default=None)
    log_group.add_argument('--logging', default='INFO', help='Logging level',
                           choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

    ######################### END #############################################
    args = parser.parse_args()

    args.exp_id = create_exp_id(args)
    logging.basicConfig(level=args.logging, format='%(asctime)s\t%(levelname)-8s\t%(message)s')
    delattr(args, 'logging')
    delattr(args, 'prefix')

    os.makedirs(args.summary_dir, exist_ok=True)
    os.makedirs(args.save_dir,    exist_ok=True)

    return args
