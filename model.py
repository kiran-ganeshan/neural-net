from optimizer import *
from layer import *
import time


def softmax(x):
    temp = np.sum(np.exp(x), 0)
    temp.shape = (1, x.shape[1])
    return np.divide(np.exp(x), np.ones((x.shape[0], 1)) @ temp)


class NeuralNetwork:

    def __init__(self, layer_sizes, act_func, filename, load_weights=False, save_weights=True, input_layer=None):
        self.output_layer = Layer(layer_sizes[0]) if input_layer is None else input_layer
        if isinstance(layer_sizes[0], tuple):
            prod = lambda tup: 1 if len(tup) == 0 else tup[0] * prod(tup[1:])
            layer_sizes[0] = prod(layer_sizes[0])
        self.layer_sizes = layer_sizes
        for i in range(1, len(layer_sizes)):
            self.output_layer = DenseLayer(layer_sizes[i], Activation.get(act_func), self.output_layer)
        self.filename = filename
        self.saving = save_weights
        if load_weights:
            self.update_weights(np.load(filename + ".npy"))
        self.weights = self.output_layer.get_weights()

    def get_weights(self):
        return self.weights

    def get_grad(self, X, y, lamb):
        diff = self.output_layer.get_grad(X, y)
        return (diff + lamb * self.weights)/X.shape[-1]

    def update_weights(self, new_weights):
        self.weights = new_weights
        self.output_layer.update_weights(new_weights)

    def predict(self, X):
        return softmax(self.output_layer.calculate(X))

    def get_cost(self, X, y, lamb):
        prediction = self.predict(X)
        error = -1 / X.shape[-1] * np.sum(np.where(y == 1, np.log(prediction), 0), axis=(0, 1))
        return error + lamb / (2 * X.shape[-1]) * np.sum(self.weights * self.weights)

    def get_accuracy(self, X, y):
        calculated = np.argmax(self.predict(X), 0)
        expected = np.argmax(y, 0)
        accuracy = sum([1 if calculated[i] == expected[i] else 0 for i in range(y.shape[-1])]) / y.shape[-1]
        return accuracy

    def save_weights(self):
        if self.saving:
            open(self.filename + ".npy", "+w")
            np.save(self.filename + ".npy", self.weights)


class ConvNeuralNetwork(NeuralNetwork):

    def __init__(self, layer_types, layer_sizes, act_func, pooling_type, filename,
                 strides=None, paddings=None, load_weights=False, save_weights=True):
        self.num_dense_layers = layer_types.count('dense')
        assert self.num_dense_layers > 0 and layer_types[0] == 'input'
        assert all([layer_type != 'dense' for layer_type in layer_types[:-self.num_dense_layers]])
        dense_layer_sizes = list(layer_sizes[-self.num_dense_layers-1:])
        layer_sizes = list(layer_sizes[0:-self.num_dense_layers])
        layer_types = list(layer_types[1:-self.num_dense_layers])
        pointer = Layer(layer_sizes[0])
        while len(layer_types) > 0:
            next_type = layer_types.pop(0)
            kwargs = dict()
            if strides is not None:
                kwargs['stride'] = strides.pop(0)
            if paddings is not None:
                kwargs['padding'] = paddings.pop(0)
            if next_type == 'conv':
                pointer = ConvLayer(layer_sizes.pop(0), Activation.get(act_func), pointer, **kwargs)
            elif next_type == 'pool':
                pointer = PoolingLayer(layer_sizes.pop(0), pooling_type, pointer, **kwargs)
        super().__init__(dense_layer_sizes, act_func, filename, load_weights, save_weights, pointer)