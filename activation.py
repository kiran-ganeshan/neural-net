import numpy as np


class Activation:

    def __init__(self, name, act, deriv, bounded=False, input_bound=None):
        self.name = name
        self.apply = act
        self.apply_deriv = deriv
        self.bounded = bounded
        self.input_bound = input_bound

    @staticmethod
    def get(act_func):
        if act_func == "sigmoid":
            return Activation("sigmoid", lambda x: np.divide(np.ones_like(x), np.ones_like(x) + np.exp(-x)),
                              lambda a: a * (np.ones_like(a) - a), True, 6)
        elif act_func == "relu":
            return Activation("relu", lambda x: np.maximum(x, 0), np.sign)
        elif act_func == "tanh":
            return Activation("tanh", np.tanh, lambda a: np.ones_like(a) - a * a, True, 4)
        elif act_func == "elu":
            rh = 1
            return Activation("elu", lambda x: np.where(x > 0, x, rh * (np.exp(x) - 1)), lambda a: np.where(a > 0, 1, a + rh))
        elif act_func == "lrelu":
            rh = 0.1
            return Activation("lrelu", lambda x: np.where(x > 0, x, rh * x), lambda a: np.where(a > 0, 1, rh))
        elif act_func == "linear":
            return Activation("linear", lambda x: x, lambda a: 1)
        elif act_func == "max":
            return Activation("max", np.max, lambda a, out: np.where(a == out, 1, 0))
        elif act_func == "avg":
            return Activation("avg", np.average, lambda a, out, weights_size_prod: 1 / (weights_size_prod * np.ones_like(a)))
        else:
            return Activation(None, None, None)
