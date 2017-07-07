import tensorflow as tf
from . import tf_extra


class Layer(object):
    '''
    Wrapper around TensorFlow layer that monitors its parameters and outputs.

    Do not use the activation parameter of any TensorFlow layer.
    Instead, use this class's activation parameter.
    That way it has access to the pre-activations.
    '''

    @classmethod
    def factory(cls, tf_layer, activation=None):
        return(
            lambda input_node: (
                cls(tf_layer, input_node, activation=activation)
            )
        )

    def __init__(self, tf_layer, input_node, activation=None):
        self.activation = activation
        self.tf_layer = tf_layer
        self.pre_activation = self.tf_layer.apply(input_node)
        if self.activation is None:
            self.post_activation = self.pre_activation
        else:
            tf.summary.histogram(
                self.name() + '/' + 'pre_activation',
                self.pre_activation
            )
            self.post_activation = self.activation(
                self.pre_activation,
                name='post_activation'
            )
        tf_extra.monitor_layer(self.tf_layer)
        tf.summary.histogram(
            self.name() + '/' + 'post_activation',
            self.post_activation
        )

    def name(self): return self.tf_layer.name
    def kernel(self): return self.tf_layer.kernel
    def bias(self): return self.tf_layer.bias
