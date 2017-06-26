import tensorflow as tf


def l1_projection_to_simplex(X, use_locking=False):
    non_negative_regrets = tf.maximum(0.0, tf.convert_to_tensor(X))
    strat = non_negative_regrets / tf.reduce_sum(non_negative_regrets, axis=0)
    return tf.where(
        tf.is_nan(strat),
        tf.fill(
            strat.shape,
            tf.reciprocal(tf.to_float(strat.shape[0]))
        ),
        strat
    )
