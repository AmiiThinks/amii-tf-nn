import os
import tensorflow as tf


class MonitoredEstimator(object):
    def __init__(self, estimator, root, *data_distributions):
        self.estimator = estimator
        self.root = root
        self.writers = {}
        self.summary_op = tf.summary.merge(
            tf.get_collection(key='summaries', scope=estimator.name_scope)
        )
        assert(self.summary_op is not None)
        for name in data_distributions:
            self.writers[name] = tf.summary.FileWriter(
                self.events_dir_base() + '-' + name
            )

    def events_dir_base(self):
        return os.path.join(self.root, self.estimator.name)
