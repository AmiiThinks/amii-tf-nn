import numpy as np
import tensorflow as tf
from . import tensorflow as tf_extra


class Layer(object):
    class Config(object):
        def __init__(
            self,
            i,
            o,
            scale=1.0,
            transfer=tf.identity,
            name='layer_1'
        ):
            self.name = name
            self.i = i
            self.o = o
            self.scale = scale
            self.transfer = transfer

        def is_compatible_with(self, other): return self.o == other.i
        def construct(self, input_tensor, seed=1):
            # Adding a name scope ensures logical grouping of the layers in the
            # graph.
            with tf.name_scope(self.name):
                with tf.name_scope('weights'):
                    weights = tf.Variable(
                        tf.truncated_normal(
                            [self.i, self.o],
                            stddev=1.0 / np.sqrt(self.i + 1),
                            seed=seed
                        )
                    )
                    tf_extra.variable_summaries(weights)
                with tf.name_scope('biases'):
                    biases = tf.Variable(tf.constant(0.1, shape=[self.o]))
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
                activations = self.transfer(preactivate, name='activation')
                tf.summary.histogram('activations', activations)
            return Layer(weights, biases, preactivate, activations)

    def __init__(self, W, b, z_hat, y_hat):
        self.W = W
        self.b = b
        self.z_hat = z_hat
        self.y_hat = y_hat

    def params(self): return [self.W, self.b]
