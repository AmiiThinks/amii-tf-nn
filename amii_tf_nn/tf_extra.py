import tensorflow as tf


def monitor_tensor(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).

    Copied from the official [TensorBoard tutorial]
    (https://www.tensorflow.org/get_started/summaries_and_tensorboard).
    """
    summaries = []
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        summaries.append(tf.summary.scalar('mean', mean))
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        summaries.append(tf.summary.scalar('stddev', stddev))
        summaries.append(tf.summary.scalar('max', tf.reduce_max(var)))
        summaries.append(tf.summary.scalar('min', tf.reduce_min(var)))
        summaries.append(tf.summary.histogram('histogram', var))
    return summaries


def monitor_layer(layer, name=''):
    summaries = None
    with tf.name_scope(name + '/' + 'kernel'):
        summaries = monitor_tensor(layer.kernel)
    if layer.bias is not None:
        with tf.name_scope(name + '/' + 'bias'):
            summaries = monitor_tensor(layer.bias)
    return summaries
