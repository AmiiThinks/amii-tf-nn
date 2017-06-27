import numpy as np
import tensorflow as tf
from . import tf_extra


class Layer(object):
    class Config(object):
        def __init__(
            self,
            i,
            o,
            training_data_inputs=None,
            training_data_targets=None,
            scale=1.0,
            transfer=tf.identity,
            name='layer_1',
            weight_init=None,
            bias_init=lambda *dims: tf.constant(0.1, shape=dims)
        ):
            self.name = name
            self.i = i
            self.o = o
            self.scale = scale
            self.transfer = transfer
            self.weight_init = weight_init
            if weight_init is None:
                self.weight_init = (
                    lambda *dims:
                        tf.truncated_normal(
                            dims,
                            stddev=1.0 / np.sqrt(i + 1)
                        )
                )

            self.bias_init = bias_init

            self.input_scaling = 1.0
            self.input_offset = 0.0
            if not(training_data_inputs is None):
                self.input_scaling = training_data_inputs.std(0)
                self.input_scaling[self.input_scaling == 0] = 1.0
                self.input_offset = training_data_inputs.mean(0)

            self.output_scaling = 1.0
            self.output_offset = 0.0
            if not(training_data_targets is None):
                self.output_scaling = training_data_targets.std(0)
                self.output_offset = training_data_targets.mean(0)

        def is_compatible_with(self, other): return self.o == other.i
        def construct(self, input_tensor):
            # Adding a name scope ensures logical grouping of the layers in the
            # graph.
            with tf.name_scope(self.name):
                with tf.name_scope('input_normalization'):
                    input_tensor = (
                        input_tensor -
                        tf.constant(self.input_offset, dtype=tf.float32)
                    ) / tf.constant(self.input_scaling, dtype=tf.float32)
                    tf.summary.histogram(
                        'inputs_after_normalization',
                        input_tensor
                    )
                with tf.name_scope('weights'):
                    weights = tf.Variable(
                        self.weight_init(self.i, self.o)
                    )
                    tf_extra.variable_summaries(weights)
                with tf.name_scope('biases'):
                    biases = tf.Variable(self.bias_init(self.o))
                    tf_extra.variable_summaries(biases)
                with tf.name_scope('scaled_Wx_plus_b'):
                    preactivate_bs = tf.matmul(input_tensor, weights) + biases
                    tf.summary.histogram(
                        'pre_activations_before_scaling',
                        preactivate_bs
                    )
                    preactivate = tf.constant(self.scale) * preactivate_bs
                    tf.summary.histogram(
                        'pre_activations',
                        preactivate
                    )
                with tf.name_scope('output_normalization'):
                    # TODO Is this the correct way to do output normalization
                    # with a non-linear transfer?
                    preactivate = (
                        preactivate *
                        tf.constant(self.output_scaling, dtype=tf.float32)
                    ) + tf.constant(self.output_offset, dtype=tf.float32)
                    tf.summary.histogram(
                        'pre_activations_after_normalization',
                        preactivate
                    )
                activations = self.transfer(preactivate, name='activation')
                tf.summary.histogram('activations', activations)
            return Layer(weights, biases, preactivate, activations)

    def __init__(self, W, b, z_hat, y_hat):
        self.W = W
        self.b = b
        self.z_hat = z_hat
        self.y_hat = y_hat

    def params(self): return [self.W, self.b]
