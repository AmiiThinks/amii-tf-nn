import tensorflow as tf


class Criterion(object):
    def __init__(self, name, node):
        self.name = name
        self.node = node
        criteria_scope = 'criteria'
        try:
            with tf.variable_scope(criteria_scope, reuse=True):
                self.variable = tf.get_variable(name, [])
        except ValueError as e:
            with tf.variable_scope(criteria_scope):
                self.variable = tf.get_variable(name, [])
            tf.summary.scalar(criteria_scope + '/' + name, self.variable)

    def run(self, sess, *args, **kwargs):
        return sess.run(self.node, *args, **kwargs)
