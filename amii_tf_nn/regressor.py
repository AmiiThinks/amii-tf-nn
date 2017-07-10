import tensorflow as tf
from .estimator import Estimator


class MseRegressor(Estimator):
    def _create_surrogate_eval(self):
        with tf.name_scope('mse'):
            mse = tf.losses.mean_squared_error(
                self.target_node,
                self.model.post_activation()
            )
        return mse, 'mse'


class HuberRegressor(Estimator):
    def _create_surrogate_eval(self):
        with tf.name_scope('huber'):
            huber = tf.losses.huber_loss(
                self.target_node,
                self.model.post_activation()
            )
        return huber, 'huber'


class MaeRegressor(Estimator):
    def _create_surrogate_eval(self):
        with tf.name_scope('mae'):
            mae = tf.losses.absolute_difference(
                self.target_node,
                self.model.post_activation()
            )
        return mae, 'mae'


class UnnormalizedEntropyLossRegressor(Estimator):
    def _create_surrogate_eval(self):
        with tf.name_scope('unnormalized_entropy_loss'):
            uel = tf.reduce_mean(
                self.model.post_activation() -
                (self.target_node * tf.log(self.model.post_activation()))
            )
        return uel, 'unnormalized_entropy_loss'
