from os import path, makedirs, getcwd
from yaml import dump, load
import tensorflow as tf
import numpy as np


class Experiment(object):
    @classmethod
    def from_yml_file(cls, path):
        with open(path, 'r') as f:
            x = cls.from_yml(f.read())
        return x

    @classmethod
    def from_yml(cls, doc):
        return cls(**load(doc))

    def __init__(
        self,
        name,
        root=getcwd(),
        seed=1,
        tag=None,
        log_level=tf.logging.ERROR
    ):
        self.root = root
        self.name = name
        self.tag = str(seed) if tag is None else str(tag)
        self.seed = seed
        tf.logging.set_verbosity(log_level)
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

    def label(self): return self.name + '_' + self.tag
    def path(self): return path.join(self.root, self.label())
    def config_file(self): return path.join(self.path(), 'config.yml')
    def exists(self):
        return path.exists(self.path()) and path.exists(self.config_file())

    def ensure_present(self):
        tf.logging.info('Saving experiment files to "{}".'.format(self.path()))
        if not path.exists(self.path()): makedirs(self.path())
        with open(self.config_file(), 'w') as f:
            dump(self.config(), stream=f, default_flow_style=False)
        tf.logging.info(
            'Run `tensorboard --logdir {}` to visualize results.'.format(
                self.path()
            )
        )
        return self

    def write_config(self):
        return self.ensure_present()

    def config(self):
        return {
            'name': self.name,
            'root': self.root,
            'seed': self.seed,
            'tag': self.tag
        }
