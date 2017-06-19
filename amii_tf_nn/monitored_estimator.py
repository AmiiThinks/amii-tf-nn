import os
import tensorflow as tf


class MonitoredEstimator(object):
    def __init__(self, estimator, root, *data_distributions):
        self.estimator = estimator
        self.root = root
        self.writers = {}
        for name in data_distributions:
            self.writers[name] = tf.summary.FileWriter(
                self.events_dir_base() + '-' + name
            )

    def events_dir_base(self):
        return os.path.join(self.root, self.estimator.name)
