import tensorflow as tf


class NetworkModel(object):
    def __init__(self, name, input_node, *layer_configs, **kwargs):
        self.name = name
        self.input_node = input_node
        self.layer_configs = layer_configs
        if len(layer_configs) < 1:
            raise(
                Exception(
                    'NetworkModel "{}" must have at least one layer.'.format(
                        self.name
                    )
                )
            )
        for i in range(len(layer_configs) - 1):
            assert(layer_configs[i].is_compatible_with(layer_configs[i + 1]))

        with tf.name_scope(self.name):
            self.layers = []
            for layer in layer_configs:
                self.layers.append(layer.construct(input_node))
                input_node = self.layers[-1].y_hat

    def pre_activation(self): return self.layers[-1].z_hat
    def post_activation(self): return self.layers[-1].y_hat
