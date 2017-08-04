import tensorflow as tf
from .monitored_estimator import MonitoredEstimator


class AbstractTrainer(object):
    def __init__(self, num_epochs=100):
        self.num_epochs = num_epochs

    def run(self):
        self._before_training()
        for i in range(self.num_epochs):
            self._before_training_epoch(i)
            self.train(i)
            self._after_training_epoch(i)
        self._after_training()


class Trainer(AbstractTrainer):
    def __init__(
        self,
        sess,
        training_data,
        *estimators,
        batches_per_epoch=None,
        **kwargs
    ):
        super(Trainer, self).__init__(**kwargs)
        self.training_data = training_data
        self.sess = sess
        self.batches_per_epoch = (
            batches_per_epoch or training_data.num_batches()
        )
        self.estimators = estimators

    def _before_training(self):
        tf.logging.info(
            'Beginning training for {} epochs.'.format(self.num_epochs)
        )

    def _after_training(self): pass
    def _before_training_epoch(self, i):
        tf.logging.info('Starting epoch {}'.format(i))

    def _after_training_epoch(self, i): pass

    def train(self, i):
        for batch, _ in self.training_data.each_batch(
            self.batches_per_epoch
        ):
            for e in self.estimators:
                self.sess.run(
                    e.optimizer,
                    feed_dict=e.to_feed_dict(batch.x, batch.y)
                )


class EvalTrainer(Trainer):
    def __init__(
        self,
        experiment_path,
        eval_data,
        *args,
        epochs_between_evaluations=1,
        **kwargs
    ):
        super(EvalTrainer, self).__init__(*args, **kwargs)
        self.epochs_between_evaluations = epochs_between_evaluations
        self.eval_data = eval_data
        self.criterion_summary_op = tf.summary.merge(
            tf.get_collection(key='summaries', scope='criteria')
        )
        self.monitored_estimators = []
        for e in self.estimators:
            self.monitored_estimators.append(
                MonitoredEstimator(
                    e,
                    experiment_path,
                    *self.eval_data.keys()
                )
            )

    def _after_training(self):
        for dist_name in self.eval_data.keys():
            self.eval(dist_name, self.num_epochs)

    def _before_training_epoch(self, i):
        tf.logging.info('Starting epoch {}'.format(i))
        if i % self.epochs_between_evaluations == 0:
            for dist_name in self.eval_data.keys():
                self.eval(dist_name, i)

    def eval(self, dist_name, i):
        num_training_steps = i * self.batches_per_epoch
        for e in self.monitored_estimators:
            batch = self.eval_data[dist_name].next_batch()
            eval_vals = e.estimator.run_evals(self.sess, batch.x, batch.y)
            summaries = self.sess.run(
                [e.summary_op, self.criterion_summary_op],
                feed_dict=e.estimator.to_feed_dict(
                    batch.x,
                    batch.y,
                    eval_vals=eval_vals
                )
            )
            for s in summaries:
                e.writers[dist_name].add_summary(s, num_training_steps)
