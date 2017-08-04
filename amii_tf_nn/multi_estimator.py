from .estimator import AbstractEstimator
from .projection import l1_projection_to_simplex


class MultiEstimator(AbstractEstimator):
    def __init__(
        self,
        base_model_factory,
        output_model_factories_and_weights,
        *args,
        **kwargs
    ):
        self.base_model_factory = base_model_factory
        self.output_model_factories_and_weights = (
            output_model_factories_and_weights
        )
        super(MultiEstimator, self).__init__(*args, **kwargs)

    def _create_model(self):
        self.base_model = self.base_model_factory()
        self.output_models = []
        self.output_model_weights = []
        for f, w in self.output_model_factories_and_weights:
            self.output_models.append(f(self.base_model.post_activation()))
            self.output_model_weights.append(w)
        self.output_model_weights = l1_projection_to_simplex(
            self.output_model_weights
        )

    def _partitioned_targets(self):
        nodes = []
        offset = 0
        for m in self.output_models:
            new_offset = offset + m.post_activation().shape[1].value
            nodes.append(self.target_node[:, offset:new_offset])
            offset = new_offset
        return nodes

    def input_node(self): return self.base_model.input_node
