from math import sqrt
from math import floor
from activation import *
from itertools import product


class Layer:

    def __init__(self, size, act_func=Activation.get(None), prev=None, weights_size=None):
        self.size = size
        if not isinstance(self.size, tuple):
            self.size = (self.size,)
        self.num_neurons = np.prod(self.size)
        self.weights_size = weights_size
        self.weights = np.array(weights_size)
        self.num_params = np.prod(self.weights_size)
        self.act = act_func
        #find multiple of n closest to f (set x=0 in lambda call)
        #checks
        #BASE CASE: if
        closest = lambda n, f, x: f + x if n % (f + x) == 0 else f - x if n % (f - x) == 0 else closest(n, f, x + 1)
        #create size tuple for layer given num_neurons and target size
        helper = lambda num, mod, close: (num,) if len(mod) < 2 else reshape_tuple(num/close, mod[:-1]) + (close,)
        reshape_tuple = lambda num, mod: helper(num, mod, closest(num, mod[-1], 0))
        if isinstance(prev, Layer):
            self.prev = prev
            self.source = None
            if len(self.size) != len(self.prev.size) and len(self.size) == 1:
                self.prev = ReshapeLayer(self.prev.num_neurons, self.prev)
            elif len(self.size) != len(self.prev.size):
                self.prev = ReshapeLayer(reshape_tuple(np.prod(self.prev.size), self.size), self.prev)
        else:
            self.prev = None
            self.source = prev
        self.logits = None

    def calculate(self, data):
        if not self.prev:
            self.logits = data if self.source is None else self.source
            return self.logits
        return self.prev.calculate(data)

    def get_weights(self):
        if self.prev:
            prev_weights = self.prev.get_weights()
            if self.weights_size:
                prev_weights = np.concatenate([self.weights.flatten(), prev_weights])
            return prev_weights
        return np.array([])

    def calculate_grad(self, delta):
        return np.array([])

    def get_grad(self, X, y):
        return self.calculate_grad(self.calculate(X) - y)

    def update_weights(self, new_weights):
        if self.prev:
            if self.weights_size:
                self.weights, new_weights = np.split(new_weights, [self.num_params])
                self.weights = np.reshape(self.weights, self.weights_size)
            self.prev.update_weights(new_weights)


class DenseLayer(Layer):

    def __init__(self, num_units, act_func, prev):
        if len(prev.size) != 1:
            prev = ReshapeLayer(prev.num_neurons, prev, act_func)
        super().__init__(num_units, act_func, prev, (num_units, prev.num_neurons + 1))
        self.weights = (2 * np.random.rand(*self.weights_size) - np.ones(self.weights_size)) / sqrt(prev.num_neurons)
        if act_func.bounded:
            self.weights *= act_func.input_bound

    def calculate(self, data):
        prev = super().calculate(data)
        concat = np.concatenate([np.ones((1,) + (prev.shape[-1],)), prev])
        self.logits = self.act.apply(self.weights @ concat)
        return self.logits

    def calculate_grad(self, delta):
        delta = delta * self.act.apply_deriv(self.logits)
        ones = np.ones((1, delta.shape[-1]))
        concat = np.transpose(np.concatenate([ones, self.prev.logits], axis=0))
        this_diff = delta @ concat
        delta = np.transpose(self.weights) @ delta
        delta = delta[1:, :]
        diff = self.prev.calculate_grad(delta)
        return np.concatenate([this_diff.flatten(), diff])


class ConvLayer(Layer):
#int is not subscriptible issue w size being used here before its created in super constructor
    def __init__(self, size, act_func, prev, stride=1, padding=((0, 0), (0, 0))):
        tool = lambda i, l: stride * (floor((prev.size[i] + padding[i][0] + padding[i][1]) / stride) - (l - 1))
        super().__init__(size, act_func, prev)
        self.weights_size = (tool(0, self.size[0]), tool(1, self.size[1]), self.prev.size[2], self.size[2])
        self.stride = stride
        for item in [padding, padding[0], padding[1]]:
            assert isinstance(item, tuple) and len(item) == 2
        padding += 2 * ((0, 0),)
        self.pad = lambda a: np.pad(a, padding)
        self.clip = lambda a: a[tuple(slice(padding[x][0], a.shape[x] - padding[x][1]) for x in range(4))]
        self.weights = 2 * np.random.rand(*self.weights_size) - np.ones(self.weights_size)

    def calculate(self, data):
        padded = self.pad(super().calculate(data))
        self.logits = np.zeros((*self.size, data.shape[-1]))
        tool = lambda x, k: slice(x * self.stride, x * self.stride + self.weights_size[k])
        for i, j in product(*map(range, self.size[:2])):
            slices = (tool(i, 0), tool(j, 1), slice(None), slice(None))
            self.logits[i, j, :, :] = np.tensordot(self.weights, padded[slices], axes=((0, 1, 2), (0, 1, 2)))
        self.logits = self.act.apply(self.logits)
        return self.logits

    def calculate_grad(self, delta):
        delta = delta * self.act.apply_deriv(self.logits)
        tool = lambda c, k: slice(c, self.stride * self.size[k] + c, self.stride)
        padded = self.pad(self.prev.logits)
        diff, prev_delta = np.zeros(self.weights_size), np.zeros(padded.shape)
        for i, j in product(*map(range, self.weights_size[:2])):
            slices = (tool(i, 0), tool(j, 1), slice(None), slice(None))
            diff[i, j, :, :] = np.tensordot(padded[slices], delta, axes=((0, 1, 3), (0, 1, 3)))
            prev_delta[slices] += np.moveaxis(np.tensordot(delta, self.weights[i, j, :, :], axes=((2,), (1,))), 2, 3)
        grad = self.prev.calculate_grad(self.clip(prev_delta))
        return np.concatenate([diff.flatten(), grad])


class PoolingLayer(Layer):

    def __init__(self, size, pooling_type, prev, stride=1, padding=((0, 0), (0, 0))):
        assert len(prev.size) == 3
        tool = lambda i, l: stride * (floor((prev.size[i] + padding[i][0] + padding[i][1]) / stride) - (l - 1))
        super().__init__(size, Activation.get(pooling_type), prev)
        padding += ((0, 0), (0, 0))
        self.window_size = (tool(0, size[0]), tool(1, size[1]))
        self.pad = lambda a: np.pad(a, padding)
        self.clip = lambda a: a[tuple([slice(padding[x][0], a.shape[x] - padding[x][1]) for x in range(4)])]
        self.tool = lambda x, k: slice(x * stride, x * stride + self.window_size[k])

    def calculate(self, data):
        padded = self.pad(super().calculate(data))
        self.logits = np.zeros(self.size + (data.shape[-1],))
        for i, j in product(*map(range, self.window_size)):
            self.logits[i, j, :, :] = self.act.apply(padded[self.tool(i, 0), self.tool(j, 1), :, :], axis=(0, 1))
        return self.logits

    def calculate_grad(self, delta):
        prev_delta = np.zeros(self.prev.size + (delta.shape[-1],))
        padded = self.pad(self.prev.logits)
        for i, j in product(*map(range, self.window_size)):
            slices = (self.tool(i, 0), self.tool(j, 1), slice(None), slice(None))
            prev_delta[slices] += self.act.apply_deriv(padded[slices], self.logits[i, j, :, :]) * delta[i, j, :, :]
        return self.prev.calculate_grad(self.clip(prev_delta))


class ReshapeLayer(Layer):

    def __init__(self, output_size, prev, dense_act_func=None):
        super().__init__(output_size, prev)
        if self.num_neurons != prev.num_neurons:
            dense_act_func = dense_act_func if dense_act_func else Activation.get("linear")
            self.prev = DenseLayer(self.num_neurons, dense_act_func, prev)

    def calculate(self, data):
        return np.reshape(super().calculate(data), self.size + (data.shape[-1],), out=self.logits)

    def calculate_grad(self, delta):
        return self.prev.calculate_grad(np.reshape(delta, self.prev.size + (delta.shape[-1],)))


class LambdaLayer(Layer):

    def __init__(self, size, op_name, act_lambda, deriv_lambda, prevs):
        assert isinstance(prevs, (list, tuple)) and act_lambda.__code__.co_argcount == len(prevs)
        super().__init__(size)
        self.act = Activation(op_name, act_lambda, deriv_lambda)
        self.prev = prevs
        self.prev_weight_sizes = None
        self.has_prev = True

    def calculate(self, data):
        super().calculate(data)
        self.logits = self.act.apply(*[x.calculate(data) for x in self.prev])
        return self.logits

    def calculate_grad(self, delta):
        lambda_gradient = self.act.apply_deriv(*[prev.logits for prev in self.prev])
        derivative, grad = list(), list()
        for prev, lamb in zip(self.prev, lambda_gradient):
            derivative.append(prev.calculate_grad(delta * lamb))
        map(grad.extend, derivative)
        return grad

    def get_weights(self):
        weights, total = [prev.get_weights() for prev in self.prev], list()
        self.prev_weight_sizes = np.array([sum(map(np.size, weight)) for weight in weights])
        map(total.extend, weights)
        return total

    def update_weights(self, new_weights):
        if self.prev_weight_sizes is None:
            self.get_weights()
        segments = np.split(new_weights, list(np.cumsum(self.prev_weight_sizes)))
        for prev, segment in zip(self.prev, segments):
            prev.update_weights(segment)
