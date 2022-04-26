import glob
import logging
import numpy as np
from numpy.linalg import inv
import os
from pprint import pformat, pprint
from scipy.stats import multivariate_normal, pearsonr
from sklearn.metrics import r2_score
import sys
import tensorflow as tf
from tensorflow.python.client import timeline

from cogsci_2019_concept_learning.utils.parse_args import parse_args
from cogsci_2019_concept_learning.utils.utils_gen import br
from cogsci_2019_concept_learning.utils.utils_plt import TensorBoardBarPlot, TensorBoardRegressionPlot, TensorBoardScatterPlot, TensorBoardActivationHeatmap
from cogsci_2019_concept_learning.utils.utils_tf import print_numeric_tensor
from cogsci_2019_concept_learning.data.dataset import Toy
from cogsci_2019_concept_learning.data.human import HumanCompDataset
from cogsci_2019_concept_learning.data.imagenet import BaselineClassification
from cogsci_2019_concept_learning.core.maml import MAMLMetaLearner
from cogsci_2019_concept_learning.core.norm import virtual_batch_norm


def get_resume_itr(fname):
    ind1 = fname.index('model_')
    resume_itr = int(fname[ind1 + len('model_'):])
    return resume_itr



def train(sess, model, train_dataset, valid_dataset, exp_id,
          train_iterations, resume, save_dir,
          summary_dir, summary_interval, save_interval, print_interval, test_print_interval,
          ):

    train_input_data = train_dataset()
    valid_input_data = valid_dataset()

    visualize = isinstance(train_dataset, Toy) and isinstance(model, MAMLMetaLearner)
    train_fetches = model.build_model(reuse=False,
                                      input_tensors=train_input_data,
                                      prefix='meta-train',
                                      train=True,
                                      return_outputs=visualize,
                                      )
    assert 'train_op' in train_fetches

    if not isinstance(train_dataset, BaselineClassification):
        valid_fetches = model.build_model(reuse=True,
                                          input_tensors=valid_input_data,
                                          prefix='meta-validation',
                                          train=False,
                                          return_outputs=visualize,
                                          )

    if summary_interval is not None:
        summ_op = tf.summary.merge_all()

        if visualize is True:

            # Add data input to fetches
            train_fetches['inputs'] = train_input_data
            valid_fetches['inputs'] = valid_input_data

            # Make plots to display on TensorBoard
            with tf.name_scope('meta-train'):
                with tf.name_scope('training'):
                    tt_plot = TensorBoardScatterPlot('data')
                    with tf.name_scope('pre-update'):
                        ttpr_plot = TensorBoardScatterPlot('prediction')
                    with tf.name_scope('post-update'):
                        ttpo_plot = TensorBoardScatterPlot('prediction')
                with tf.name_scope('validation'):
                    tv_plot = TensorBoardScatterPlot('data')
                    with tf.name_scope('pre-update'):
                        tvpr_plot = TensorBoardScatterPlot('prediction')
                    with tf.name_scope('post-update'):
                        tvpo_plot = TensorBoardScatterPlot('prediction')

            with tf.name_scope('meta-validation'):
                with tf.name_scope('pre-update'):
                    vvpr_density_plot = TensorBoardActivationHeatmap('activation')
                with tf.name_scope('post-update'):
                    vvpo_density_plot = TensorBoardActivationHeatmap('activation')
                with tf.name_scope('training'):
                    vt_plot = TensorBoardScatterPlot('data')
                    with tf.name_scope('pre-update'):
                        vtpr_plot = TensorBoardScatterPlot('prediction')
                    with tf.name_scope('post-update'):
                        vtpo_plot = TensorBoardScatterPlot('prediction')
                with tf.name_scope('validation'):
                    vv_plot = TensorBoardScatterPlot('data')
                    with tf.name_scope('pre-update'):
                        vvpr_pred_plot = TensorBoardScatterPlot('prediction')
                    with tf.name_scope('post-update'):
                        vvpo_pred_plot = TensorBoardScatterPlot('prediction')

            valid_summaries = tf.summary.merge([
                vt_plot.summary,
                vv_plot.summary,
                vtpr_plot.summary,
                vtpo_plot.summary,
                vvpr_pred_plot.summary,
                vvpo_pred_plot.summary,
                vvpr_density_plot.summary,
                vvpo_density_plot.summary,
            ])
            train_summaries = tf.summary.merge([
                tt_plot.summary,
                tv_plot.summary,
                ttpr_plot.summary,
                tvpr_plot.summary,
                ttpo_plot.summary,
                tvpo_plot.summary,
            ])

    # Commence training / testing
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), keep_checkpoint_every_n_hours=0.2)

    resume_itr = 0

    if resume_itr == 0:
        sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Finalize the graph. The graph is read-only after this statement.
    sess.graph.finalize()

    if summary_interval is not None:
        summary_writer = tf.summary.FileWriter(os.path.join(summary_dir, exp_id), sess.graph)

    run_metadata = tf.RunMetadata()
    trace_file = open('/src/timeline.ctf.json', 'w')

    try:

        for itr in range(resume_itr, train_iterations):

            summary, train_result = sess.run([summ_op, train_fetches], options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)

            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            trace_file.write(trace.generate_chrome_trace_format())
            trace_file.close()
            import pdb; pdb.set_trace()

            if print_interval is not None and itr != 0 and itr % print_interval == 0:

                logging.info('Train iteration %d' % (itr))
                print_output(train_result)

            if not isinstance(train_dataset, BaselineClassification) and test_print_interval is not None and itr != 0 and itr % test_print_interval == 0:

                valid_result = sess.run(valid_fetches)

                logging.info('====== Validation ==========================')
                print_output(valid_result)

                if summary_interval is not None and itr % summary_interval == 0:

                    # Validation visualization
                    if visualize is True:

                        plot_args = {
                            'x_min': -valid_dataset.grid_bounds,
                            'x_max': valid_dataset.grid_bounds,
                            'y_min': -valid_dataset.grid_bounds,
                            'y_max': valid_dataset.grid_bounds,
                            'vmin':  0.,
                            'vmax':  1.,
                        }

                        # Have to slice out data prediction from density visualization
                        split_idx = -valid_dataset.grid_ticks**2
                        vvpr_pred    = np.squeeze(valid_result['outputs']['pre-update valid'][:,  :split_idx, :], axis=-1)
                        vvpr_density = np.squeeze(valid_result['outputs']['pre-update valid'][:,  split_idx:, :], axis=-1)
                        vvpo_pred    = np.squeeze(valid_result['outputs']['post-update valid'][:, :split_idx, :], axis=-1)
                        vvpo_density = np.squeeze(valid_result['outputs']['post-update valid'][:, split_idx:, :], axis=-1)

                        x = valid_result['inputs']['validation data'][:, :split_idx, 0]
                        y = valid_result['inputs']['validation data'][:, :split_idx, 1]
                        vvpr_pred_plot_summary = vvpr_pred_plot.plot(x, y, vvpr_pred, **plot_args)

                        x = valid_result['inputs']['validation data'][:, :split_idx, 0]
                        y = valid_result['inputs']['validation data'][:, :split_idx, 1]
                        vvpo_pred_plot_summary = vvpo_pred_plot.plot(x, y, vvpo_pred, **plot_args)

                        # Activation heatmap visualization
                        width = int(np.sqrt(vvpr_density.shape[1]))
                        vvpr_density = np.reshape(vvpr_density, [-1, width, width])

                        width = int(np.sqrt(vvpo_density.shape[1]))
                        vvpo_density = np.reshape(vvpo_density, [-1, width, width])

                        vvpr_density_plot_summary = vvpr_density_plot.plot(vvpr_density, vmin=0., vmax=1.)
                        vvpo_density_plot_summary = vvpo_density_plot.plot(vvpo_density, vmin=0., vmax=1.)

                        # Data scatter plots
                        x = valid_result['inputs']['training data'][:, :, 0]
                        y = valid_result['inputs']['training data'][:, :, 1]
                        z = np.squeeze(valid_result['inputs']['training labels'], axis=-1)
                        vt_plot_summary = vt_plot.plot(x, y, z, **plot_args)

                        x = valid_result['inputs']['validation data'][:, :split_idx, 0]
                        y = valid_result['inputs']['validation data'][:, :split_idx, 1]
                        z = np.squeeze(valid_result['inputs']['validation labels'][:, :split_idx, :], axis=-1)
                        vv_plot_summary = vv_plot.plot(x, y, z, **plot_args)

                        # Prediction scatter plots
                        x = valid_result['inputs']['training data'][:, :, 0]
                        y = valid_result['inputs']['training data'][:, :, 1]
                        z = np.squeeze(valid_result['outputs']['pre-update train'], axis=-1)
                        vtpr_plot_summary = vtpr_plot.plot(x, y, z, **plot_args)

                        x = valid_result['inputs']['training data'][:, :, 0]
                        y = valid_result['inputs']['training data'][:, :, 1]
                        z = np.squeeze(valid_result['outputs']['post-update train'], axis=-1)
                        vtpo_plot_summary = vtpo_plot.plot(x, y, z, **plot_args)

                        feed_dict = {
                            vt_plot.placeholder:            vt_plot_summary,
                            vv_plot.placeholder:            vv_plot_summary,
                            vtpr_plot.placeholder:          vtpr_plot_summary,
                            vtpo_plot.placeholder:          vtpo_plot_summary,
                            vvpr_pred_plot.placeholder:     vvpr_pred_plot_summary,
                            vvpo_pred_plot.placeholder:     vvpo_pred_plot_summary,
                            vvpr_density_plot.placeholder:  vvpr_density_plot_summary,
                            vvpo_density_plot.placeholder:  vvpo_density_plot_summary,
                        }
                        summary_writer.add_summary(sess.run(valid_summaries, feed_dict=feed_dict), itr)

            if summary_interval is not None and itr % summary_interval == 0:
                summary_writer.add_summary(summary, itr)

                # Training visualization
                if visualize is True:

                    plot_args = {
                        'x_min': -valid_dataset.grid_bounds,
                        'x_max': valid_dataset.grid_bounds,
                        'y_min': -valid_dataset.grid_bounds,
                        'y_max': valid_dataset.grid_bounds,
                        'vmin':  0.,
                        'vmax':  1.,
                    }

                    # Data scatter plots
                    x = train_result['inputs']['training data'][:, :, 0]
                    y = train_result['inputs']['training data'][:, :, 1]
                    z = np.squeeze(train_result['inputs']['training labels'], axis=-1)
                    tt_plot_summary = tt_plot.plot(x, y, z, **plot_args)

                    x = train_result['inputs']['validation data'][:, :, 0]
                    y = train_result['inputs']['validation data'][:, :, 1]
                    z = np.squeeze(train_result['inputs']['validation labels'], axis=-1)
                    tv_plot_summary = tv_plot.plot(x, y, z, **plot_args)

                    # Prediction scatter plots
                    x = train_result['inputs']['training data'][:, :, 0]
                    y = train_result['inputs']['training data'][:, :, 1]
                    z = np.squeeze(train_result['outputs']['pre-update train'], axis=-1)
                    ttpr_plot_summary = ttpr_plot.plot(x, y, z, **plot_args)

                    x = train_result['inputs']['training data'][:, :, 0]
                    y = train_result['inputs']['training data'][:, :, 1]
                    z = np.squeeze(train_result['outputs']['post-update train'], axis=-1)
                    ttpo_plot_summary = ttpo_plot.plot(x, y, z, **plot_args)

                    x = train_result['inputs']['validation data'][:, :, 0]
                    y = train_result['inputs']['validation data'][:, :, 1]
                    z = np.squeeze(train_result['outputs']['pre-update valid'], axis=-1)
                    tvpr_plot_summary = tvpr_plot.plot(x, y, z, **plot_args)

                    x = train_result['inputs']['validation data'][:, :, 0]
                    y = train_result['inputs']['validation data'][:, :, 1]
                    z = np.squeeze(train_result['outputs']['post-update valid'], axis=-1)
                    tvpo_plot_summary = tvpo_plot.plot(x, y, z, **plot_args)

                    feed_dict = {
                        tt_plot.placeholder:   tt_plot_summary,
                        tv_plot.placeholder:   tv_plot_summary,
                        ttpr_plot.placeholder: ttpr_plot_summary,
                        tvpr_plot.placeholder: tvpr_plot_summary,
                        ttpo_plot.placeholder: ttpo_plot_summary,
                        tvpo_plot.placeholder: tvpo_plot_summary,
                    }
                    summary_writer.add_summary(sess.run(train_summaries, feed_dict=feed_dict), itr)

            if save_interval is not None and itr != 0 and itr % save_interval == 0:
                logging.info('\t\tSaving model at iteration %d...' % (itr))
                if not os.path.isdir(os.path.join(save_dir, exp_id)):
                    os.mkdir(os.path.join(save_dir, exp_id))
                saver.save(sess, os.path.join(save_dir, exp_id, 'model_%d' % itr), write_meta_graph=False, write_state=False)

    except Exception as e:
        # Report exceptions to the coordinator.
        coord.request_stop()
        logging.error(str(e))

    finally:
        # Terminate as usual. It is safe to call `coord.request_stop()` twice.
        coord.request_stop()
        coord.join(threads)


def test(sess, model, is_train, valid_dataset, test_dataset, exp_id, save_dir, summary_dir, num_val_batches, num_test_batches):

    # Choose the best parameters if the validation dataset is provided
    if valid_dataset is not None:
        valid_input_data = valid_dataset()
        valid_fetches = model.build_model(reuse=is_train,
                                          input_tensors=valid_input_data,
                                          prefix=None,
                                          train=False,
                                          test=True,
                                          )

    # Test set
    test_input_data = test_dataset()
    test_fetches = model.build_model(reuse=is_train or valid_dataset is not None,
                                     input_tensors=test_input_data,
                                     prefix='meta-test',
                                     train=False,
                                     test=True,
                                     )

    summ_op = tf.summary.merge_all()

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=6, keep_checkpoint_every_n_hours=0.5)

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    model_file = None
    if valid_dataset is not None:

        model_files = [x[:-6] for x in glob.glob(os.path.join(save_dir, exp_id, '*.index'))]

        if model_files:

            best_val_acc = -np.inf
            best_model_file = model_files[0]

            logging.info('Mean validation accuracy, stddev, and confidence intervals:')
            for model_file in model_files:
                resume_itr = get_resume_itr(model_file)
                saver.restore(sess, model_file)

                val_accuracies = []
                for _ in range(num_val_batches):
                    try:
                        acc_fetch = valid_fetches['accuracies']['post-update valid']
                    except KeyError:
                        acc_fetch = valid_fetches['total accuracy']
                    acc = sess.run(acc_fetch)
                    val_accuracies.append(acc)

                val_accuracies = np.array(val_accuracies)
                mean = np.mean(val_accuracies)
                std = np.std(val_accuracies)
                ci95 = 1.96 * std / np.sqrt(num_val_batches)

                logging.info('itr %04d: %.4f (%.4f), [%.4f, %.4f]' % (resume_itr, mean, std, mean - ci95, mean + ci95))

                if np.mean(val_accuracies) > best_val_acc:
                    best_val_acc = np.mean(val_accuracies)
                    best_model_file = model_file

            model_file = best_model_file

    else:
        # Just load the latest checkpoint
        model_files = [x[:-6] for x in glob.glob(os.path.join(save_dir, exp_id, '*.index'))]
        model_file = model_files[np.argmax([get_resume_itr(model_file) for model_file in model_files])]

    if model_file:
        resume_itr = get_resume_itr(model_file)
        logging.info("Restoring model weights from %s at iteration %d for testing." % (model_file, resume_itr))
        saver.restore(sess, model_file)
    else:
        logging.error('Failed to find model file to resume/test from.')
        raise Exception

    if summary_dir is not None:
        summary_writer = tf.summary.FileWriter(os.path.join(summary_dir, exp_id), sess.graph)

    for itr in range(num_test_batches):
        summary = sess.run(summ_op)
        summary_writer.add_summary(summary, itr)

    logging.info("Finished evaluation.")


def human_comp(sess, model, human_comp_dataset, is_train, is_test, exp_id, save_dir, summary_dir):

    # Graph construction
    human_input_data_placeholders = {
        'training data':     tf.placeholder(tf.float32, [1, None] + human_comp_dataset.shape_input),
        'training labels':   tf.placeholder(tf.float32, [1, None, 1]),
        'validation data':   tf.placeholder(tf.float32, [1, None] + human_comp_dataset.shape_input),
        'validation labels': tf.placeholder(tf.float32, [1, None, 1]),
    }

    human_comp_fetches = model.build_model(reuse=is_train or is_test,
                                           input_tensors=human_input_data_placeholders,
                                           prefix='human-comp',
                                           train=False,
                                           test=True,
                                           return_outputs=True,
                                           )

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=6)
    resume_itr = 0
    model_file = None
    model_file = tf.train.latest_checkpoint(os.path.join(save_dir, exp_id))

    plots = {}
    with tf.name_scope('human-comp'):
        for metric in ['model_binary', 'model_point_prediction', 'human_response', 'point_prediction_fit', 'binary_fit']:
            plots[metric] = {}
            plots['%s_norm' % metric] = {}
            with tf.name_scope(metric):
                for cond in human_comp_dataset.conditions:
                    with tf.name_scope(cond):
                        if 'fit' in metric:
                            plots[metric][cond] = TensorBoardRegressionPlot('human_model_comp')
                            plots['%s_norm' % metric][cond] = TensorBoardRegressionPlot('human_model_comp_normed')
                        else:
                            plots[metric][cond] = TensorBoardBarPlot('gen_prob')

    sess.run(tf.group(tf.local_variables_initializer(), tf.global_variables_initializer()))
    if model_file:
        resume_itr = get_resume_itr(model_file)
        logging.info("Restoring model weights from %s." % model_file)
        saver.restore(sess, model_file)
    else:
        logging.error('Failed to find model file to resume/test from.')
        raise Exception

    # Initialize results dictionary
    def init_response_dict():
        results = {}
        for cond in human_comp_dataset.conditions:
            results[cond] = {}
            for num_ex in human_comp_dataset.num_training_samples:
                results[cond][num_ex] = {}
                for match_type in human_comp_dataset.match_types:
                    results[cond][num_ex][match_type] = []
        return results

    # Loop over experimental conditions in the human data
    model_responses = init_response_dict()
    human_responses = init_response_dict()
    for condition in human_comp_dataset.conditions:
        for num_training_samples in human_comp_dataset.num_training_samples:
            for batch in human_comp_dataset(num_training_samples, condition):

                # Model response
                train_data      = batch['training data']
                train_labels    = batch['training labels']
                valid_data      = batch['validation data']
                response_labels = batch['response labels']
                gt_labels       = batch['gt labels']

                feed_dict = {
                    human_input_data_placeholders['training data']:     train_data,
                    human_input_data_placeholders['training labels']:   train_labels,
                    human_input_data_placeholders['validation data']:   valid_data,
                    human_input_data_placeholders['validation labels']: response_labels,
                }
                outputs = sess.run(human_comp_fetches, feed_dict)
                model_response = outputs['outputs']['post-update valid']

                for j, match_type in enumerate(human_comp_dataset.match_types):
                    model_responses[condition][num_training_samples][match_type] += [model_response[np.where(gt_labels == j)]]
                    human_responses[condition][num_training_samples][match_type] += [response_labels[np.where(gt_labels == j)]]

    # Initialize results dictionary
    def init_results_dict():
        results = {}
        for cond in human_comp_dataset.conditions:
            results[cond] = {}
            for metric in ['mean', 'std', 'conf']:
                results[cond][metric] = [[] for _ in range(len(human_comp_dataset.match_types))]
        return results

    # Define metrics
    def mean(x):
        return np.mean(x) if x.size != 0 else 0.

    def std(x):
        return np.std(x) if x.size != 0 else 0.

    def conf(x):
        """Return the delta of the 95% confidence interval for the mean of x."""
        return np.std(x) * 1.96 / np.sqrt(np.prod(x.shape)) if x.size != 0 else 0.

    # Collapse results
    pp_results = init_results_dict()
    bin_results = init_results_dict()
    human_results = init_results_dict()

    for condition in human_comp_dataset.conditions:
        for i, match_type in enumerate(human_comp_dataset.match_types):
            for num_ex in human_comp_dataset.num_training_samples:

                model_responses[condition][num_ex][match_type] = np.reshape(np.array(model_responses[condition][num_ex][match_type]), -1)
                human_responses[condition][num_ex][match_type] = np.reshape(np.array(human_responses[condition][num_ex][match_type]), -1)

                for metric_lbl, metric_func in zip(['mean', 'std', 'conf'], [mean, std, conf]):

                    model_res = model_responses[condition][num_ex][match_type]
                    pp_results[condition][metric_lbl][i]  += [metric_func(model_res)]
                    bin_results[condition][metric_lbl][i] += [metric_func(np.round(model_res))]

                    human_res = human_responses[condition][num_ex][match_type]
                    human_results[condition][metric_lbl][i] += [metric_func(human_res)]

    # Plots
    plot_summaries = []
    feed_dict = {}

    # Bar plots
    for metric, results in zip(['model_binary', 'model_point_prediction', 'human_response'], [bin_results, pp_results, human_results]):
        for condition in human_comp_dataset.conditions:
            plot = plots[metric][condition]
            plot_summary = plot.plot(np.array(results[condition]['mean']),
                                     np.array(results[condition]['conf']),
                                     human_comp_dataset.num_training_samples,
                                     human_comp_dataset.match_types,
                                     y_min=0., y_max=1.,
                                     )
            feed_dict[plot.placeholder] = plot_summary
            plot_summaries += [plot.summary]

    # Regression plots
    to_correlate_x = []
    to_correlate_y = []
    for metric, results in zip(['binary_fit', 'point_prediction_fit'], [bin_results, pp_results]):
        for condition in human_comp_dataset.conditions:

            # Unnormalized
            plot = plots[metric][condition]

            model_res = np.reshape(results[condition]['mean'], -1)
            human_res = np.reshape(human_results[condition]['mean'], -1)

            plot_summary = plot.plot(model_res, human_res,
                                     x_min=0., x_max=1.,
                                     y_min=0., y_max=1.,
                                     )
            feed_dict[plot.placeholder] = plot_summary
            plot_summaries += [plot.summary]

            # Normalized
            plot = plots['%s_norm' % metric][condition]

            def norm_by_sub_match(x):
                x /= x[0, :]
                return np.nan_to_num(x)

            model_res = np.reshape(norm_by_sub_match(np.array(results[condition]['mean'])), -1)
            human_res = np.reshape(norm_by_sub_match(np.array(human_results[condition]['mean'])), -1)

            plot_summary = plot.plot(model_res, human_res,
                                     x_min=0., x_max=1.,
                                     y_min=0., y_max=1.,
                                     )
            feed_dict[plot.placeholder] = plot_summary
            plot_summaries += [plot.summary]

            to_correlate_x += [model_res]
            to_correlate_y += [human_res]

    plot_summaries = tf.summary.merge(plot_summaries)
    summary_writer = tf.summary.FileWriter(os.path.join(summary_dir, exp_id), sess.graph)
    summary_writer.add_summary(sess.run(plot_summaries, feed_dict=feed_dict))

    to_correlate_x = np.reshape(to_correlate_x, [-1])
    to_correlate_y = np.reshape(to_correlate_y, [-1])
    pearson_corr = pearsonr(to_correlate_x, to_correlate_y)
    r_squared    = r2_score(to_correlate_x, to_correlate_y)

    logging.info("Finished human comparison.")
    logging.info("Pearson correlation between human and model results is " + str(pearson_corr))
    logging.info("Coefficient of determination between human and model results is " + str(r_squared))


def main(args=sys.argv[1:]):

    args = vars(parse_args(args))

    logging.info('Parameter settings:')
    logging.info(pformat(args, indent=4, width=1, depth=1))

    # Some params for dataset creation
    train_dataset_cls            = args['train_dataset'];     del args['train_dataset']
    val_dataset_cls              = args['val_dataset'];       del args['val_dataset']
    test_dataset_cls             = args['test_dataset'];      del args['test_dataset']
    data_source_dir              = args['data_source_dir'];   del args['data_source_dir']
    input_size                   = args['input_size'];        del args['input_size']
    num_workers                  = args['num_workers'];       del args['num_workers']
    is_train                     = args['train'];             del args['train']
    is_test                      = args['test'];              del args['test']
    is_human_comp                = args['human_comp'];        del args['human_comp']

    num_train_batches            = args['num_train_batches']; del args['num_train_batches']
    num_val_batches              = args['num_val_batches'];   del args['num_val_batches']
    num_test_batches             = args['num_test_batches'];  del args['num_test_batches']
    num_classes                  = args['num_classes']
    update_training_batch_size   = args['update_training_batch_size']
    update_validation_batch_size = args['update_validation_batch_size']
    meta_batch_size              = args['meta_batch_size']

    # Get training params
    resume =                  args['resume'];                     del args['resume']
    exp_id =                  args['exp_id'];                     del args['exp_id']
    num_train_iters =         args['num_train_iters'];            del args['num_train_iters']
    summary_dir =             args['summary_dir'];                del args['summary_dir']
    summary_interval =        args['summary_interval'];           del args['summary_interval']
    save_dir =                args['save_dir'];                   del args['save_dir']
    save_interval =           args['save_interval'];              del args['save_interval']
    print_interval =          args['print_interval'];             del args['print_interval']
    test_print_interval =     args['test_print_interval'];        del args['test_print_interval']
    random_seed =             args['random_seed'];                del args['random_seed']
    use_reference_batch =     args['norm'] == virtual_batch_norm

    model_cls = args['model_arch']; del args['model_arch']

    if not os.path.isdir(os.path.join(summary_dir, exp_id)):
        os.mkdir(os.path.join(summary_dir, exp_id))

    if is_train:
        train_dataset = train_dataset_cls(num_training_samples=update_training_batch_size,
                                          num_validation_samples=update_validation_batch_size,
                                          meta_batch_size=meta_batch_size,
                                          source_dir=data_source_dir,
                                          output_dir=os.path.join(summary_dir, exp_id),
                                          input_size=input_size,
                                          num_classes=num_classes,
                                          num_total_batches=num_train_batches,
                                          random_seed=random_seed,
                                          mode='train',
                                          num_workers=num_workers,
                                          use_reference_batch=use_reference_batch,
                                          )

    val_dataset = val_dataset_cls(num_training_samples=update_training_batch_size,
                                  num_validation_samples=update_validation_batch_size,
                                  meta_batch_size=1,
                                  source_dir=data_source_dir,
                                  output_dir=os.path.join(summary_dir, exp_id),
                                  input_size=input_size,
                                  num_classes=num_classes,
                                  num_total_batches=num_val_batches,
                                  random_seed=random_seed,
                                  mode='val',
                                  num_workers=num_workers,
                                  use_reference_batch=use_reference_batch,
                                  )

    if is_test:
        test_dataset = test_dataset_cls(num_training_samples=update_training_batch_size,
                                        num_validation_samples=update_validation_batch_size,
                                        meta_batch_size=1,
                                        source_dir=data_source_dir,
                                        output_dir=os.path.join(summary_dir, exp_id),
                                        input_size=input_size,
                                        num_classes=num_classes,
                                        num_total_batches=num_test_batches,
                                        random_seed=random_seed,
                                        mode='test',
                                        num_workers=num_workers,
                                        use_reference_batch=use_reference_batch,
                                        )

    if is_human_comp:
        human_comp_dataset = HumanCompDataset(input_size=input_size, random_seed=random_seed)

    # Create model
    model_params = args.copy()
    model_params['input_shape'] = [input_size, input_size, 3]
    model_params['size_reference_batch'] = 10  # TODO: modularize

    model = model_cls(**model_params)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # do not use all the space on a GPU
    with tf.Session(config=config) as sess:

        if is_train:
            train(sess, model, train_dataset, val_dataset, exp_id, num_train_iters, resume, save_dir,
                  summary_dir, summary_interval, save_interval, print_interval, test_print_interval)

        if is_test:
            test(sess, model, is_train, None, test_dataset, exp_id, save_dir, summary_dir, num_val_batches, num_test_batches)

        if is_human_comp:
            human_comp(sess, model, human_comp_dataset, is_train, is_test, exp_id, save_dir, summary_dir)


if __name__ == "__main__":
    sys.exit(main())
