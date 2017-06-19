from amii_tf_nn.data import DataStream
import numpy as np


def test_next():
    x = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([[1, 2], [3, 4], [5, 6]])

    patient = DataStream(x, y)
    assert len(patient.next(1)) == 1
    assert len(patient.next(1)) == 1
    assert len(patient.next(1)) == 1
    assert len(patient.next(1)) == 1

    patient = DataStream(x, y)
    assert len(patient.next(2)) == 2
    assert len(patient.next(2)) == 2

    patient = DataStream(x, y)
    d = patient.next(4)
    assert len(d) == 4

    patient = DataStream(x, y)
    d = patient.next(5)
    assert len(d) == 5

    patient = DataStream(x, y)
    d = patient.next(6)
    assert len(d) == 6

    patient = DataStream(x, y)
    d = patient.next(7)
    assert len(d) == 7
