from amii_tf_nn.projection import l1_projection_to_simplex
import tensorflow as tf
import pytest


def test_l1_no_negative():
    patient = l1_projection_to_simplex(tf.constant([2.0, 8.0, 0.0]))
    with tf.Session() as sess:
        print(sess.run(patient))
        strat = sess.run(patient)
        x_strat = [0.2, 0.8, 0.0]
        assert len(strat) == len(x_strat)
        for i in range(len(strat)):
            assert strat[i] == pytest.approx(x_strat[i])


def test_l1_with_negative():
    patient = l1_projection_to_simplex(tf.constant([2.0, 8.0, -5.0]))
    with tf.Session() as sess:
        print(sess.run(patient))
        strat = sess.run(patient)
        x_strat = [0.2, 0.8, 0.0]
        assert len(strat) == len(x_strat)
        for i in range(len(strat)):
            assert strat[i] == pytest.approx(x_strat[i])
