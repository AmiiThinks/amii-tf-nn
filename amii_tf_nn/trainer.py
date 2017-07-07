import tensorflow as tf
from .monitored_estimator import MonitoredEstimator


class _Trainer(object):
    def __init__(self, data, num_epochs=100):
        self.data = data
        self.num_epochs = num_epochs

    def run(self):
        self._before_training()
        for i in range(self.num_epochs):
            self._before_training_epoch(i)
            self.train(i)
            self._after_training_epoch(i)
        self._after_training()


class Trainer(_Trainer):
    def __init__(
        self,
        sess,
        data,
        *estimators,
        batches_per_epoch=None,
        **kwargs
    ):
        super(Trainer, self).__init__(data, **kwargs)
        self.sess = sess
        self.batches_per_epoch = batches_per_epoch
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
        for batch, _ in self.data['training'].each_batch(
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
        *args,
        epochs_between_evaluations=1,
        batches_per_eval=None,
        **kwargs
    ):
        super(EvalTrainer, self).__init__(*args, **kwargs)
        self.epochs_between_evaluations = epochs_between_evaluations
        self.batches_per_eval = batches_per_eval
        self.criterion_summary_op = tf.summary.merge(
            tf.get_collection(key='summaries', scope='criteria')
        )
        self.monitored_estimators = []
        for e in self.estimators:
            self.monitored_estimators.append(
                MonitoredEstimator(e, experiment_path, *self.data.keys())
            )

    def _after_training(self):
        for dist_name in self.data.keys():
            self.eval(dist_name, self.num_epochs)

    def _before_training_epoch(self, i):
        tf.logging.info('Starting epoch {}'.format(i))
        if i % self.epochs_between_evaluations == 0:
            for dist_name in self.data.keys():
                self.eval(dist_name, i)

    def eval(self, dist_name, i):
        for e in self.monitored_estimators:
            for batch, j in self.data[dist_name].each_batch(
                self.batches_per_eval
            ):
                surrogate_eval = e.estimator.run_surrogate_eval(
                    self.sess,
                    batch.x,
                    batch.y
                )
                eval_vals = e.estimator.run_evals(self.sess, batch.x, batch.y)
                summaries = self.sess.run(
                    [e.summary_op, self.criterion_summary_op],
                    feed_dict=e.estimator.to_feed_dict(
                        batch.x,
                        batch.y,
                        surrogate_eval=surrogate_eval,
                        eval_vals=eval_vals
                    )
                )
                for s in summaries:
                    e.writers[dist_name].add_summary(
                        s,
                        i * self.batches_per_epoch + j
                    )
