import tensorflow as tf


def monitor_layer(layer):
    summaries = [
        tf.summary.histogram('kernel', layer.kernel)
    ]
    if layer.bias is not None:
        summaries.append(tf.summary.histogram('bias', layer.bias))
    return summaries
