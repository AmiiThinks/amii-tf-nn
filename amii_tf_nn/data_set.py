class DataSet(object):
    def __init__(self, **data):
        self.data = data
        num_features = None
        num_outputs = None
        for k in self.keys():
            if num_features is None: num_features = self[k].num_features()
            assert(num_features == self[k].num_features())

            if num_outputs is None: num_outputs = self[k].num_outputs()
            assert(num_outputs == self[k].num_outputs())

    def __contains__(self, k): return k in self.data
    def __len__(self): return len(self.data)
    def __getitem__(self, k): return self.data[k]
    def __setitem__(self, k, v):
        assert(self.num_features() == v.num_features())
        assert(self.num_outputs() == v.num_outputs())
        self.data[k] = v
    def __missing__(self, k): raise(KeyError('No data named "{}".'.format(k)))
    def keys(self): return self.data.keys()
    def values(self): return self.data.values()
    def items(self): return self.data.items()

    def num_features(self):
        return next(iter(self.data.values())).num_features()

    def num_outputs(self):
        return next(iter(self.data.values())).num_outputs()
