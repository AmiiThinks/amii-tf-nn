import tensorflow as tf
from .monitored_estimator import MonitoredEstimator


class _Trainer(object):
    def __init__(
        self,
        data,
        num_epochs=100,
        batch_size=100,
        epochs_between_evaluations=10,
        batches_per_epoch=None
    ):
        self.data = data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.epochs_between_evaluations = epochs_between_evaluations
        self.batches_per_epoch = batches_per_epoch

    def run(self):
        for i in range(self.num_epochs):
            if i % self.epochs_between_evaluations == 0:
                for dist_name in self.data.keys():
                    self.eval(dist_name, i)
            self.train(i)
        for dist_name in self.data.keys():
            self.eval(dist_name, self.num_epochs)


class Trainer(_Trainer):
    def __init__(self, sess, experiment, data, *estimators, **kwargs):
        super(Trainer, self).__init__(data, **kwargs)
        self.sess = sess
        self.merged_summary = tf.summary.merge_all()
        self.combined_events_writer = tf.summary.FileWriter(
            experiment.path(),
            sess.graph
        )
        self.monitored_estimators = []
        for e in estimators:
            self.monitored_estimators.append(
                MonitoredEstimator(e, experiment.path(), *data.keys())
            )

    def train(self, i):
        for batch, _ in self.data['training'].each_batch(
            self.batches_per_epoch
        ):
            for e in self.monitored_estimators:
                self.sess.run(
                    e.estimator.optimizer,
                    feed_dict=e.estimator.to_feed_dict(batch.x, batch.y)
                )

    def eval(self, dist_name, i):
        for e in self.monitored_estimators:
            for batch, j in self.data[dist_name].each_batch(1):
                surrogate_eval = e.estimator.run_surrogate_eval(
                    self.sess,
                    batch.x,
                    batch.y
                )
                eval_vals = e.estimator.run_evals(self.sess, batch.x, batch.y)
                summary = self.sess.run(
                    self.merged_summary,
                    feed_dict=e.estimator.to_feed_dict(
                        batch.x,
                        batch.y,
                        surrogate_eval=surrogate_eval,
                        eval_vals=eval_vals
                    )
                )
                e.writers[dist_name].add_summary(
                    summary,
                    i * self.batches_per_epoch + j
                )
