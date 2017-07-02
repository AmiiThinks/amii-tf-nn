import tensorflow as tf
from .criterion import Criterion


class Estimator(object):
    def __init__(
        self,
        name,
        model_factory,
        target_node,
        **optimization_params
    ):
        self.name = name
        self.target_node = target_node
        self.optimization_params = optimization_params
        with tf.name_scope(self.name):
            self.name_scope = tf.contrib.framework.get_name_scope()
            self.model = model_factory()
            surrogate_eval_node, surrogate_eval_name =\
                self._create_surrogate_eval()
            self.optimizer = self._create_optimizer(surrogate_eval_node)
            eval_nodes = self._create_evals()
        self.surrogate_criterion = Criterion(
            surrogate_eval_name,
            surrogate_eval_node
        )
        self.eval_criteria = []
        for name in eval_nodes.keys():
            self.eval_criteria.append(Criterion(name, eval_nodes[name]))

    def run_surrogate_eval(self, sess, x, y):
        return self.surrogate_criterion.run(
            sess,
            feed_dict={
                self.model.input_node: x,
                self.target_node: y
            }
        )

    def run_evals(self, sess, x, y):
        return [
            criterion.run(
                sess,
                feed_dict={
                    self.model.input_node: x,
                    self.target_node: y
                }
            ) for criterion in self.eval_criteria
        ]

    def to_feed_dict(
        self,
        x,
        y,
        surrogate_eval=None,
        eval_vals=None
    ):
        d = {
            self.model.input_node: x,
            self.target_node: y
        }
        if not(surrogate_eval is None):
            d[self.surrogate_criterion.variable] = surrogate_eval
        if not(eval_vals is None):
            for i in range(len(eval_vals)):
                d[self.eval_criteria[i].variable] = eval_vals[i]
        return d
