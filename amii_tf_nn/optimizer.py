import tensorflow as tf


class AdamOptimizerMixin(object):
    def _create_optimizer(self, surrogate_eval_node):
        with tf.name_scope('adam_training'):
            node = tf.train.AdamOptimizer(
                **self.optimization_params
            ).minimize(surrogate_eval_node)
        return node
