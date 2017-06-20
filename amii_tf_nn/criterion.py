from . import tf_extra


class Criterion(object):
    def __init__(self, name, node):
        self.name = name
        self.variable = tf_extra.criterion_variable(name)
        self.node = node

    def run(self, sess, *args, **kwargs):
        return sess.run(self.node, *args, **kwargs)
