import tensorflow as tf


def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).

    Copied from the official [TensorBoard tutorial]
    (https://www.tensorflow.org/get_started/summaries_and_tensorboard).
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def criterion_variable(name):
    v_name = name + '_mean'
    try:
        with tf.variable_scope(name):
            v = tf.get_variable(v_name, [])
        tf.summary.scalar(name, v)
    except ValueError as e:
        with tf.variable_scope(name, reuse=True):
            v = tf.get_variable(v_name, [])
    return v
