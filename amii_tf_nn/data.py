from __future__ import division, generators
import numpy as np
from math import ceil


class Data(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        assert(self.x.shape[0] == self.y.shape[0])

    def __len__(self): return self.x.shape[0]
    def __getitem__(self, slice_): return Data(self.x[slice_], self.y[slice_])
    def __add__(self, other):
        return Data(
            np.concatenate((self.x, other.x)),
            np.concatenate((self.y, other.y))
        )
    def num_features(self): return self.x.shape[1]
    def num_outputs(self): return self.y.shape[1]


class DataStream(Data):
    @classmethod
    def from_data(cls, data): return cls(data.x, data.y)

    def __init__(self, *args, **kwargs):
        super(DataStream, self).__init__(*args, **kwargs)
        self.i = 0

    def next(self, n):
        old_i = self.i
        self.i = (self.i + n) % len(self)
        if n + old_i >= len(self):
            d = self[old_i:]
            for _ in range((n - len(d)) // len(self)):
                d += self
            d += self[:self.i]
            assert(len(d) == n)
            return d
        else:
            return self[old_i:self.i]


class BatchedData(DataStream):
    @classmethod
    def from_data(cls, batch_size, data):
        return cls(batch_size, data.x, data.y)

    def __init__(self, batch_size, *args, **kwargs):
        super(BatchedData, self).__init__(*args, **kwargs)
        self.batch_size = batch_size

    def each_batch(self, num_batches=None):
        if num_batches is None: num_batches = self.num_batches()
        for i in range(num_batches): yield self.next_batch(), i

    def next_batch(self, i=None):
        if i is not None: self.i = i
        return self.next(self.batch_size)

    def num_batches(self):
        return ceil(len(self) / self.batch_size)
