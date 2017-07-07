import tensorflow as tf


def monitor_layer(layer):
    summaries = [
        tf.summary.histogram(layer.name + '/' + 'kernel', layer.kernel)
    ]
    if layer.bias is not None:
        summaries.append(
            tf.summary.histogram(layer.name + '/' + 'bias', layer.bias)
        )
    return summaries
