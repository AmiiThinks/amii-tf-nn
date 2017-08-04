from __future__ import division, generators
import numpy as np
from math import ceil


class Data(object):
    def __init__(self, x, y):
        self.data_has_been_copied = False
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
    def __str__(self):
        s = ''
        for i in range(len(self)):
            s += '{} | {}'.format(self.x[i, :], self.y[i, :])
        return s

    def num_features(self):
        try:
            n = self.x.shape[1]
        except IndexError:
            n = 1
        return n

    def num_outputs(self):
        try:
            n = self.y.shape[1]
        except IndexError:
            n = 1
        return n

    def shuffle(self):
        if not self.data_has_been_copied:
            self.x = self.x.copy()
            self.y = self.y.copy()
            self.data_has_been_copied = True
        rng_state = np.random.get_state()
        np.random.shuffle(self.x)
        np.random.set_state(rng_state)
        np.random.shuffle(self.y)
        return self


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
    def from_data(cls, data, batch_size=None, **kwargs):
        return cls(data.x, data.y, batch_size=batch_size, **kwargs)

    def __init__(self, *args, batch_size=None, **kwargs):
        super(BatchedData, self).__init__(*args, **kwargs)
        if batch_size is None:
            self.batch_size = len(self)
        else:
            self.batch_size = max(1, min(batch_size, len(self)))

    def each_batch(self, num_batches=None):
        if num_batches is None: num_batches = self.num_batches()
        for i in range(num_batches): yield self.next_batch(), i

    def next_batch(self, i=None):
        if i is not None: self.i = i
        return self.next(self.batch_size)

    def num_batches(self):
        return ceil(len(self) / self.batch_size)


class ShuffledBatchedData(BatchedData):
    def each_batch(self, *args, **kwargs):
        self.shuffle()
        return super(ShuffledBatchedData, self).each_batch(*args, **kwargs)
