import tensorflow as tf


class NetworkModel(object):
    @classmethod
    def factory(cls, *args, **kwargs):
        return lambda: cls(*args, **kwargs)

    def __init__(self, name, input_node, *layer_factories, **kwargs):
        self.name = name
        self.input_node = input_node

        if len(layer_factories) < 1:
            raise(
                Exception(
                    'NetworkModel "{}" must have at least one layer.'.format(
                        self.name
                    )
                )
            )
        with tf.name_scope(self.name):
            self.layers = []
            for layer in layer_factories:
                self.layers.append(layer(input_node))
                input_node = self.layers[-1].post_activation

    def pre_activation(self): return self.layers[-1].pre_activation
    def post_activation(self): return self.layers[-1].post_activation
