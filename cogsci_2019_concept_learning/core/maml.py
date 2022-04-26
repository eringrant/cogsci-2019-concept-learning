import numpy as np
import tensorflow as tf

from cogsci_2019_concept_learning.core.model import *
#from cogsci_2019_concept_learning.core.special_grads import _MaxPoolGradGrad, _SoftplusGradGrad  # uncomment if not using bleeding edge TensorFlow

from cogsci_2019_concept_learning.utils.utils_gen import br, log_function_call
from cogsci_2019_concept_learning.utils.utils_tf import print_numeric_tensor, cosine_distance, euclidean_distance


class MAMLMetaLearner(object):

    def __init__(self,
                 input_shape,
                 num_classes,
                 init,
                 norm,
                 nonlin,
                 meta_step_size,
                 update_step_size,
                 meta_batch_size,
                 update_training_batch_size,
                 update_validation_batch_size,
                 num_updates,
                 meta_clip_norm,
                 update_clip_norm,
                 meta_optimizer,
                 update_optimizer,
                 size_reference_batch,
                 *args,
                 **kwargs):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_updates = num_updates

        self.meta_step_size   = tf.placeholder_with_default(meta_step_size,   (), name="meta_step_size")
        self.update_step_size = tf.placeholder_with_default(update_step_size, (), name="update_step_size") if update_step_size else None

        self.meta_batch_size              = meta_batch_size
        self.update_training_batch_size   = update_training_batch_size
        self.update_validation_batch_size = update_validation_batch_size

        self.meta_clip_norm   = meta_clip_norm
        self.update_clip_norm = update_clip_norm

        self.meta_optimizer = meta_optimizer
        self.update_rule    = update_optimizer

        self.size_reference_batch = size_reference_batch
        self.norm = norm
        self.forward_pass = self.forward_pass_cls(init, nonlin, norm, num_classes, size_reference_batch)  #TODO: norm unused
        self.accuracy = self.accuracy_cls()
        self.per_class_accuracy = self.per_class_accuracy_cls()
        self.loss_func = self.loss_func_cls()

    """
    def forward_pass(self, *args, **kwargs):
        raise NotImplementedError("Abstract method")

    def loss_func(self, *args, **kwargs):
        raise NotImplementedError("Abstract method")

    def accuracy(self, *args, **kwargs):
        raise NotImplementedError("Abstract method")
    """

    def construct_weights(self, scope, reuse):

        with tf.variable_scope(scope, reuse=reuse):
            batch_size = self.update_training_batch_size
            if self.norm == virtual_batch_norm:
                batch_size += self.size_reference_batch
            dummy_input = tf.zeros([batch_size] + self.input_shape, name="dummy_input")
            self.forward_pass(dummy_input, True, None)

        # Create a dictionary of weights from the collection corresponding to the provided scope
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        weights = {w.name.lstrip('%s' % scope)[1:-2]: w for w in weights}

        return weights

    def build_model(self, reuse, input_tensors, train, prefix, scope='model', return_outputs=False, test=False):

        weights = self.construct_weights(scope, reuse=reuse)
        self.variable_list = list(weights.keys())  # ordered list to pull out gradients

        inputa = input_tensors['training data']
        labela = input_tensors['training labels']
        inputb = input_tensors['validation data']
        labelb = input_tensors['validation labels']

        with tf.variable_scope(scope, reuse=True):

            update_step_size = self.update_step_size

            @log_function_call("map of MAML batch meta-learn function")
            def batch_metalearn(inp):
                """a for train, b for test

                Arguments:
                    inputa: A [meta_batch_size, num_classes*samples_per_class, dim_input] Tensor.
                    labela: A [meta_batch_size, num_classes*samples_per_class, num_classes] Tensor.
                    inputb: A [meta_batch_size, num_classes*samples_per_class, dim_input] Tensor.
                    labelb: A [meta_batch_size, num_classes*samples_per_class, num_classes] Tensor.
                """
                local_inputa, local_inputb, local_labela, local_labelb = inp
                labelb_pos, labelb_neg = tf.split(local_labelb, 2, axis=0) #TODO assumes ordering of validation batch!!

                local_outputs_a_pre,  local_outputs_b_pre = [], []
                local_outputs_a_post, local_outputs_b_post = [], []
                local_losses_a_pre,   local_accuracies_a_pre = [], []
                local_losses_a_post,  local_accuracies_a_post = [], []
                local_losses_b_pre,   local_accuracies_b_pre = [], []
                local_losses_b_post,  local_accuracies_b_post = [], []

                local_per_class_accuracies_b_pre, local_per_class_accuracies_b_post = [], []

                gradient_list = []

                fast_weights = weights  # start at iteration 0 with the original weights
                fast_states  = {var: None for var in fast_weights}  # start at iteration 0 with the null state

                for j in range(self.num_updates):

                    # Forward prop through elementary training data (before gradient update steps)
                    output_a_pre   = self.forward_pass(local_inputa, train, fast_weights)
                    loss_a_pre     = self.loss_func(output_a_pre, local_labela)
                    #loss_a_pre = tf.Print(loss_a_pre, [loss_a_pre], message='apre')
                    accuracy_a_pre = self.accuracy(output_a_pre,  local_labela)

                    # Forward prop through elementary validation data (before gradient update steps)
                    output_b_pre   = self.forward_pass(local_inputb, train, fast_weights)
                    loss_b_pre     = self.loss_func(output_b_pre, local_labelb)
                    accuracy_b_pre = self.accuracy(output_b_pre,  local_labelb)
                    per_class_accuracy_b_pre = self.per_class_accuracy(output_b_pre, local_labelb)

                    # Compute gradients of training loss wrt fast weights
                    gradients = dict(zip(fast_weights.keys(), tf.gradients(loss_a_pre, list(fast_weights.values()), name="update_gradients")))

                    # Clip gradients if necessary
                    if self.update_clip_norm:
                        gradients = {var: tf.clip_by_norm(grad, self.update_clip_norm, name="update_grad_clip_%s" % var) for var, grad in gradients.items()}

                    # Perform backward pass
                    new_fast_weights = {}
                    new_fast_states  = {}
                    for key in gradients:
                        update, state = self.update_rule(gradients=gradients[key], state=fast_states[key], learning_rate=update_step_size)
                        new_fast_weights[key] = fast_weights[key] + update
                        new_fast_states[key]  = state
                    fast_weights = new_fast_weights
                    fast_states  = new_fast_states

                    # Forward prop through elementary training data (after gradient update steps)
                    output_a_post   = self.forward_pass(local_inputa, train, fast_weights)
                    loss_a_post     = self.loss_func(output_a_post, local_labela)
                    accuracy_a_post = self.accuracy(output_a_post,  local_labela)

                    # Forward prop through elementary validation data (after gradient update steps)
                    output_b_post   = self.forward_pass(local_inputb, train, fast_weights)
                    loss_b_post     = self.loss_func(output_b_post, local_labelb)
                    #loss_b_post = tf.Print(loss_b_post, [loss_b_post], message='bpost')
                    accuracy_b_post = self.accuracy(output_b_post,  local_labelb)
                    per_class_accuracy_b_post = self.per_class_accuracy(output_b_post, local_labelb)

                    local_outputs_a_pre     += [output_a_pre] if not isinstance(self.forward_pass, GenerativeConvNet) else [output_a_pre[1]]
                    local_losses_a_pre      += [loss_a_pre]
                    local_outputs_b_pre     += [output_b_pre] if not isinstance(self.forward_pass, GenerativeConvNet) else [output_b_pre[1]]
                    local_losses_b_pre      += [loss_b_pre]
                    local_accuracies_a_pre  += [accuracy_a_pre]
                    local_accuracies_b_pre  += [accuracy_b_pre]
                    local_outputs_a_post    += [output_a_post] if not isinstance(self.forward_pass, GenerativeConvNet) else [output_a_post[1]]
                    local_losses_a_post     += [loss_a_post]
                    local_outputs_b_post    += [output_b_post] if not isinstance(self.forward_pass, GenerativeConvNet) else [output_b_post[1]]
                    local_losses_b_post     += [loss_b_post]
                    local_accuracies_a_post += [accuracy_a_post]
                    local_accuracies_b_post += [accuracy_b_post]
                    local_per_class_accuracies_b_pre += [per_class_accuracy_b_pre]
                    local_per_class_accuracies_b_post += [per_class_accuracy_b_post]
                    gradient_list           += [list([gradients[k] for k in self.variable_list])]

                local_fn_output =  []
                #local_fn_output += [local_outputs_a_pre,  local_outputs_b_pre,  local_losses_a_pre,  local_losses_b_pre,  local_accuracies_a_pre,  local_accuracies_b_pre]
                #local_fn_output += [local_outputs_a_post, local_outputs_b_post, local_losses_a_post, local_losses_b_post, local_accuracies_a_post, local_accuracies_b_post]
                #local_fn_output += [local_per_class_accuracies_b_pre, local_per_class_accuracies_b_post]
                #local_fn_output += [gradient_list]
                local_fn_output += tf.gradients(local_losses_b_post[self.num_updates - 1], [weights[key] for key in self.variable_list])

                return local_fn_output

            num_loop_variables = len(weights)
            #out_dtype =  [[tf.float32] * self.num_updates] * 14                    # loss & accuracy types
            #out_dtype += [[[tf.float32] * num_loop_variables] * self.num_updates]  # gradient types
            #out_dtype = [[[tf.float32] * num_loop_variables] * self.num_updates]  # gradient types
            out_dtype = [tf.float32] * num_loop_variables  # gradient types
            meta_batch_size = int(inputa.get_shape().as_list()[0])

            result = tf.map_fn(batch_metalearn, elems=(inputa, inputb, labela, labelb), dtype=out_dtype, parallel_iterations=meta_batch_size, name="batch_metalearn")
            #outputs_a_pre, outputs_b_pre, losses_a_pre, losses_b_pre, accuracies_a_pre, accuracies_b_pre, \
                #outputs_a_post, outputs_b_post, losses_a_post, losses_b_post, accuracies_a_post, accuracies_b_post, \
                #per_class_accuracies_b_pre, per_class_accuracies_b_post, \
                #gradient_list = result

            gradient_list = result

            total_losses_a_pre = [tf.constant(0)] #[tf.reduce_mean(losses_a_pre[j]) for j in range(self.num_updates)]
            total_losses_b_pre = [tf.constant(0)] #[tf.reduce_mean(losses_b_pre[j]) for j in range(self.num_updates)]

            total_losses_a_post = [tf.constant(0)] #[tf.reduce_mean(losses_a_post[j]) for j in range(self.num_updates)]
            total_losses_b_post = [tf.constant(0)] #[tf.reduce_mean(losses_b_post[j]) for j in range(self.num_updates)]

            total_accuracies_a_pre = [tf.constant(0)] #[tf.reduce_mean(accuracies_a_pre[j]) for j in range(self.num_updates)]
            total_accuracies_b_pre = [tf.constant(0)] #[tf.reduce_mean(accuracies_b_pre[j]) for j in range(self.num_updates)]

            total_accuracies_a_post = [tf.constant(0)] #[tf.reduce_mean(accuracies_a_post[j]) for j in range(self.num_updates)]
            total_accuracies_b_post = [tf.constant(0)] #[tf.reduce_mean(accuracies_b_post[j]) for j in range(self.num_updates)]

            total_per_class_accuracies_b_pre  = [tf.constant(0)] #[tf.reduce_mean(per_class_accuracies_b_pre[j],  axis=[0]) for j in range(self.num_updates)]
            total_per_class_accuracies_b_post = [tf.constant(0)] #[tf.reduce_mean(per_class_accuracies_b_post[j], axis=[0]) for j in range(self.num_updates)]

            tensor_outputs = {
                'losses': {
                    'train pre':  total_losses_a_pre[0],
                    'train post': total_losses_a_post[-1],
                    'valid pre':  total_losses_b_pre[0],
                    'valid post': total_losses_b_post[-1],
                },
                'accuracies': {
                    'train pre':            total_accuracies_a_pre[0],
                    'train post':           total_accuracies_a_post[-1],
                    'valid pre':            total_accuracies_b_pre[0],
                    'valid post':           total_accuracies_b_post[-1],
                    'valid post per-class':  total_per_class_accuracies_b_pre[0],
                    'valid post per-class': total_per_class_accuracies_b_post[-1],
                },
            }

            if return_outputs:

                tensor_outputs['outputs'] = {
                    'pre-update train':  tf.nn.sigmoid(outputs_a_pre[0]),
                    'post-update train': tf.nn.sigmoid(outputs_a_post[-1]),
                    'pre-update valid':  tf.nn.sigmoid(outputs_b_pre[0]),
                    'post-update valid': tf.nn.sigmoid(outputs_b_post[-1]),
                }

            # Summaries
            for j in range(self.num_updates):

                tf.summary.scalar('%s/pre-update_train_loss/step_%d'      % (prefix, j + 1), total_losses_a_pre[j])
                tf.summary.scalar('%s/pre-update_train_accuracy/step_%d'  % (prefix, j + 1), total_accuracies_a_pre[j])

                tf.summary.scalar('%s/pre-update_valid_loss/step_%d'       % (prefix, j + 1), total_losses_b_pre[j])
                tf.summary.scalar('%s/pre-update_valid_accuracy/step_%d'   % (prefix, j + 1), total_accuracies_b_pre[j])

                tf.summary.scalar('%s/post-update_train_loss/step_%d'     % (prefix, j + 1), total_losses_a_post[j])
                tf.summary.scalar('%s/post-update_train_accuracy/step_%d' % (prefix, j + 1), total_accuracies_a_post[j])

                tf.summary.scalar('%s/post-update_valid_loss/step_%d'      % (prefix, j + 1), total_losses_b_post[j])
                tf.summary.scalar('%s/post-update_valid_accuracy/step_%d'  % (prefix, j + 1), total_accuracies_b_post[j])

                #for i in range(max(self.num_classes, 2)):
                #    tf.summary.scalar('%s/pre-update_valid_accuracy_class_%d/step_%d' % (prefix, i, j + 1), total_per_class_accuracies_b_pre[j][i])
                #    tf.summary.scalar('%s/post-update_valid_accuracy_class_%d/step_%d' % (prefix, i, j + 1), total_per_class_accuracies_b_post[j][i])

                #for var_name, grad in zip(self.variable_list, gradient_list[j]):
                #    tf.summary.histogram('%s_inner_loop_gradients/step_%d/%s' % (prefix, j + 1, var_name.replace(":", "_")), grad)

        if train:

            #train_loss = total_losses_b_post[self.num_updates - 1]  # backprop all the way from the end of the inner loop
            #optimizer = self.meta_optimizer(self.meta_step_size)
            #train_gvs = optimizer.compute_gradients(train_loss, list(weights.values()))

            #if self.meta_clip_norm:
                #train_gvs = [(tf.clip_by_norm(grad, self.meta_clip_norm, name="meta_grad_clip_%s" % var.name.replace("%s/" % scope, "").replace("/", "_").replace(":0", "")), var) for grad, var in train_gvs]

            gradient_list = [tf.reduce_mean(gradient, axis=0) for gradient in gradient_list]
            train_gvs = [(gradient_list[i], sorted(tf.trainable_variables(), key=lambda x: self.variable_list.index(x.name[6:-2]))[i]) for i in range(len(self.variable_list))]
            optimizer = tf.train.GradientDescentOptimizer(0.1)
            train_op = optimizer.apply_gradients(train_gvs)
            tensor_outputs['train_op'] = train_op

            # Summaries
            #for grad, var in train_gvs:
            #    var_name = '/'.join(var.name.split('/')[-2:])
            #    tf.summary.histogram('train_variables/%s'      % var_name.replace(":", "_"), var)
            #    tf.summary.histogram('train_meta_gradients/%s' % var_name.replace(":", "_"), grad)

        return tensor_outputs


class SmallOneLayerFullyConnectedBinaryMAMLMetaLearner(MAMLMetaLearner):

    name = 'small_one_layer_fc_binary_maml'
    forward_pass_cls = SmallOneLayerFullyConnectedNetwork
    accuracy_cls = BinaryClassificationAccuracy
    per_class_accuracy_cls = BinaryPerClassClassificationAccuracy
    loss_func_cls = BinaryClassificationLoss


class LargeOneLayerFullyConnectedBinaryMAMLMetaLearner(MAMLMetaLearner):

    name = 'large_one_layer_fc_binary_maml'
    forward_pass_cls = LargeOneLayerFullyConnectedNetwork
    accuracy_cls = BinaryClassificationAccuracy
    per_class_accuracy_cls = BinaryPerClassClassificationAccuracy
    loss_func_cls = BinaryClassificationLoss


class SmallTwoLayerFullyConnectedBinaryMAMLMetaLearner(MAMLMetaLearner):

    name = 'small_two_layer_fc_binary_maml'
    forward_pass_cls = SmallTwoLayerFullyConnectedNetwork
    accuracy_cls = BinaryClassificationAccuracy
    per_class_accuracy_cls = BinaryPerClassClassificationAccuracy
    loss_func_cls = BinaryClassificationLoss


class LargeTwoLayerFullyConnectedBinaryMAMLMetaLearner(MAMLMetaLearner):

    name = 'large_two_layer_fc_binary_maml'
    forward_pass_cls = LargeTwoLayerFullyConnectedNetwork
    accuracy_cls = BinaryClassificationAccuracy
    per_class_accuracy_cls = BinaryPerClassClassificationAccuracy
    loss_func_cls = BinaryClassificationLoss


class SmallOneLayerFullyConnectedMultiClassMAMLMetaLearner(MAMLMetaLearner):

    name = 'small_one_layer_fc_multiclass_maml'
    forward_pass_cls = SmallOneLayerFullyConnectedNetwork
    accuracy_cls = MultiClassClassificationAccuracy
    per_class_accuracy_cls = MultiClassPerClassClassificationAccuracy
    loss_func_cls = MultiClassClassificationLoss


class LargeOneLayerFullyConnectedMultiClassMAMLMetaLearner(MAMLMetaLearner):

    name = 'large_one_layer_fc_multiclass_maml'
    forward_pass_cls = LargeOneLayerFullyConnectedNetwork
    accuracy_cls = MultiClassClassificationAccuracy
    per_class_accuracy_cls = MultiClassPerClassClassificationAccuracy
    loss_func_cls = MultiClassClassificationLoss


class SmallTwoLayerFullyConnectedMultiClassMAMLMetaLearner(MAMLMetaLearner):

    name = 'small_two_layer_fc_multiclass_maml'
    forward_pass_cls = SmallTwoLayerFullyConnectedNetwork
    accuracy_cls = MultiClassClassificationAccuracy
    per_class_accuracy_cls = MultiClassPerClassClassificationAccuracy
    loss_func_cls = MultiClassClassificationLoss


class LargeTwoLayerFullyConnectedMultiClassMAMLMetaLearner(MAMLMetaLearner):

    name = 'large_two_layer_fc_multiclass_maml'
    forward_pass_cls = LargeTwoLayerFullyConnectedNetwork
    accuracy_cls = MultiClassClassificationAccuracy
    per_class_accuracy_cls = MultiClassPerClassClassificationAccuracy
    loss_func_cls = MultiClassClassificationLoss


class SmallConvBinaryMAMLMetaLearner(MAMLMetaLearner):

    name = 'small_conv_binary_maml'
    forward_pass_cls = SmallConvNet
    accuracy_cls = BinaryClassificationAccuracy
    per_class_accuracy_cls = BinaryPerClassClassificationAccuracy
    loss_func_cls = BinaryClassificationLoss


class LargeConvBinaryMAMLMetaLearner(MAMLMetaLearner):

    name = 'large_conv_binary_maml'
    forward_pass_cls = LargeConvNet
    accuracy_cls = BinaryClassificationAccuracy
    per_class_accuracy_cls = BinaryPerClassClassificationAccuracy
    loss_func_cls = BinaryClassificationLoss


class SmallMaxPoolConvBinaryMAMLMetaLearner(MAMLMetaLearner):

    name = 'small_maxpoolconv_binary_maml'
    forward_pass_cls = SmallMaxPoolConvNet
    accuracy_cls = BinaryClassificationAccuracy
    per_class_accuracy_cls = BinaryPerClassClassificationAccuracy
    loss_func_cls = BinaryClassificationLoss


class LargeMaxPoolConvBinaryMAMLMetaLearner(MAMLMetaLearner):

    name = 'large_maxpoolconv_binary_maml'
    forward_pass_cls = LargeMaxPoolConvNet
    accuracy_cls = BinaryClassificationAccuracy
    per_class_accuracy_cls = BinaryPerClassClassificationAccuracy
    loss_func_cls = BinaryClassificationLoss


class SmallConvMultiClassMAMLMetaLearner(MAMLMetaLearner):

    name = 'small_conv_multiclass_maml'
    forward_pass_cls = SmallConvNet
    accuracy_cls = MultiClassClassificationAccuracy
    per_class_accuracy_cls = MultiClassPerClassClassificationAccuracy
    loss_func_cls = MultiClassClassificationLoss


class LargeConvMultiClassMAMLMetaLearner(MAMLMetaLearner):

    name = 'large_conv_multiclass_maml'
    forward_pass_cls = LargeConvNet
    accuracy_cls = MultiClassClassificationAccuracy
    per_class_accuracy_cls = MultiClassPerClassClassificationAccuracy
    loss_func_cls = MultiClassClassificationLoss


class SmallMaxPoolConvMultiClassMAMLMetaLearner(MAMLMetaLearner):

    name = 'small_maxpoolconv_multiclass_maml'
    forward_pass_cls = SmallMaxPoolConvNet
    accuracy_cls = MultiClassClassificationAccuracy
    per_class_accuracy_cls = MultiClassPerClassClassificationAccuracy
    loss_func_cls = MultiClassClassificationLoss


class LargeMaxPoolConvMultiClassMAMLMetaLearner(MAMLMetaLearner):

    name = 'large_maxpoolconv_multiclass_maml'
    forward_pass_cls = LargeMaxPoolConvNet
    accuracy_cls = MultiClassClassificationAccuracy
    per_class_accuracy_cls = MultiClassPerClassClassificationAccuracy
    loss_func_cls = MultiClassClassificationLoss

class LargeConvGenerativeMAMLMetaLearner(MAMLMetaLearner):

    name = 'large_conv_generative_maml'
    forward_pass_cls = LargeGenerativeConvNet
    accuracy_cls = NormalDistributionAccuracy
    per_class_accuracy_cls = NormalDistributionAccuracy
    loss_func_cls = NormalDistributionLoss
    
