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
            transfer=tf.identity,
            name='layer_1',
            weight_init=None,
            bias_init=lambda *dims: tf.constant(0.1, shape=dims)
        ):
            self.name = name
            self.i = i
            self.o = o
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
                # TODO
                self.input_scaling = training_data_inputs.std(0)
                self.input_scaling[self.input_scaling == 0] = 1.0
                self.input_offset = training_data_inputs.mean(0)

            self.output_scaling = 1.0
            self.output_offset = 0.0
            if not(training_data_targets is None):
                # TODO
                self.output_scaling = training_data_targets.std(0)
                self.output_offset = training_data_targets.mean(0)

        def is_compatible_with(self, other): return self.o == other.i

        def construct(self, input_tensor, seed=1):
            # Adding a name scope ensures logical grouping of the layers in the
            # graph.
            with tf.name_scope(self.name):
                input_tensor = self._normalize_inputs(input_tensor)
                preactivate = self._normalize_outputs(
                    self._create_preactivations(input_tensor, seed)
                )
                activations = self.transfer(preactivate, name='activation')
                tf.summary.histogram('activations', activations)
            return Layer(preactivate, activations)

        def _normalize_outputs(self, preactivate):
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
            return preactivate

    class FullyConnectedConfig(Config):
        def _normalize_inputs(self, input_tensor):
            with tf.name_scope('input_normalization'):
                input_tensor = (
                    input_tensor -
                    tf.constant(self.input_offset, dtype=tf.float32)
                ) / tf.constant(self.input_scaling, dtype=tf.float32)
                tf.summary.histogram(
                    'inputs_after_normalization',
                    input_tensor
                )
            return input_tensor

        def _create_preactivations(self, input_tensor, seed):
            preactivate_bs = tf.layers.dense(
                input_tensor,
                self.o,
                use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(
                    seed=seed
                ),
                bias_initializer=tf.zeros_initializer(),
                name=self.name
            )
            preactivate = tf.constant(self.scale) * preactivate_bs
            tf.summary.histogram('pre_activations', preactivate)
            return preactivate

    class ConvolutionalConfig(Config):
        # TODO
        # def _normalize_inputs(self, input_tensor):
        #     with tf.name_scope('input_normalization'):
        #         input_tensor = (
        #             input_tensor -
        #             tf.constant(self.input_offset, dtype=tf.float32)
        #         ) / tf.constant(self.input_scaling, dtype=tf.float32)
        #         tf.summary.histogram(
        #             'inputs_after_normalization',
        #             input_tensor
        #         )
        #     return input_tensor
        #
        # def _create_weights(self, seed=1):
        #     with tf.name_scope('weights'):
        #         weights = tf.Variable(
        #             tf.truncated_normal(
        #                 [self.i, self.o],
        #                 stddev=1.0 / np.sqrt(self.i + 1),
        #                 seed=seed
        #             )
        #         )
        #         tf_extra.variable_summaries(weights)
        #     return weights

        def _create_preactivations(self, weights, biases, input_tensor):
            with tf.name_scope('convolution'):
                preactivate_bs = (
                    nn_ops.conv2d(
                        input_tensor,
                        weights,
                        strides=[1, 1, 1, 1],
                        padding="SAME"
                    ) +
                    biases
                )
                tf.summary.histogram(
                    'pre_activations_before_scaling',
                    preactivate_bs
                )
                preactivate = tf.constant(self.scale) * preactivate_bs
                tf.summary.histogram(
                    'pre_activations',
                    preactivate
                )
            return preactivate

    def __init__(self, z_hat, y_hat):
        self.z_hat = z_hat
        self.y_hat = y_hat
