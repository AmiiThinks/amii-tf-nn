import tensorflow as tf
from .criterion import Criterion


class AbstractEstimator(object):
    def __init__(
        self,
        name,
        target_node,
        **optimization_params
    ):
        self.name = name
        self.target_node = target_node
        self.optimization_params = optimization_params
        with tf.name_scope(self.name):
            self.name_scope = tf.contrib.framework.get_name_scope()
            self._create_model()
            surrogate_eval_node, surrogate_eval_name =\
                self._create_surrogate_eval()
            self.optimizer = self._create_optimizer(surrogate_eval_node)
            eval_nodes = self._create_evals()
        self.surrogate_criterion = Criterion(
            surrogate_eval_name,
            surrogate_eval_node
        )
        self.eval_criteria = [self.surrogate_criterion]
        if eval_nodes is not None:
            for name in eval_nodes.keys():
                self.eval_criteria.append(Criterion(name, eval_nodes[name]))

    def _create_evals(self): return None

    def run_evals(self, sess, x, y):
        return sess.run(
            [criterion.node for criterion in self.eval_criteria],
            feed_dict={
                self.input_node(): x,
                self.target_node: y
            }
        )

    def to_feed_dict(
        self,
        x,
        y,
        eval_vals=None
    ):
        d = {
            self.input_node(): x,
            self.target_node: y
        }
        if not(eval_vals is None):
            for i in range(len(eval_vals)):
                d[self.eval_criteria[i].variable] = eval_vals[i]
        return d


class Estimator(AbstractEstimator):
    def __init__(
        self,
        model_factory,
        *args,
        **kwargs
    ):
        self.model_factory = model_factory
        super(Estimator, self).__init__(*args, **kwargs)

    def _create_model(self): self.model = self.model_factory()
    def input_node(self): return self.model.input_node
