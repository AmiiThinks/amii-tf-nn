import tensorflow as tf


def l1_projection_to_simplex(tensor, row_normalize=False):
    non_negative = tf.maximum(0.0, tensor)
    if row_normalize:
        non_negative = tf.transpose(non_negative)
    fallback = 1.0 / non_negative.shape[0].value
    projected = non_negative / tf.reduce_sum(non_negative, axis=0)
    if row_normalize:
        projected = tf.transpose(projected)
    return tf.where(
        tf.is_nan(projected),
        tf.zeros_like(projected) + tf.to_float(fallback),
        projected
    )
