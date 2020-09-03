import random
from layer import *
import numpy as np
from unittest import TestCase
__name__ = "Layer Tests"


class LayerTest(TestCase):

    def setUp(self):
        num_ex = random.randint(3, 20)
        fake_num_ex = random.randint(3, 20)
        data = np.random.randint(-10, 10, (4, 4, 3, num_ex))
        fake_data = np.random.randint(-20, -11, (4, 4, 3, fake_num_ex))
        labels = np.random.randint(-10, 10, (5, num_ex))
        bottom = Layer((4, 4, 3), Activation.get(None), fake_data)
        mid = Layer((4, 4, 3), Activation.get(None), bottom)
        top = Layer(5, Activation.get(None), mid)
        self.set_values(
            layers=[bottom, mid, top],
            data=data,
            labels=labels,
            sizes=[(4, 4, 3), (4, 4, 3), (5,)]
        )

    def set_values(self, layers, sizes, data=None, labels=None, weight_sizes=None, act_names=None, logits=None,
                   output=None, weights=None, grad=None, update=None, new_weights=None, prevs=None):
        self.layers = layers
        self.data = data
        self.labels = labels
        self.exp_sizes = sizes
        self.exp_prevs = [None] + layers[:-1] if prevs is None else prevs
        self.exp_weight_sizes = len(layers) * [None] if weight_sizes is None else weight_sizes
        self.exp_act_names = len(layers) * [None] if act_names is None else act_names
        default_logits = lambda lay: lay.source if lay.source is not None else data if not lay.prev else None
        self.exp_logits = [default_logits(layer) for layer in layers] if logits is None else logits
        self.exp_output = output if output is not None else None if logits is None else logits[-1]
        self.exp_weights = weights if weights is not None else []
        self.exp_grad = grad
        self.new_weights = update
        self.exp_new_weights = new_weights

    def test_init(self):
        self.assertEqual(self.exp_sizes, [x.size for x in self.layers], "Sizes incorrect")
        self.assertEqual(self.exp_prevs, [x.prev for x in self.layers], "Prevs incorrect")
        self.assertEqual(self.exp_weight_sizes, [x.weights_size for x in self.layers], "Weight sizes incorrect")
        self.assertEqual(self.exp_act_names, [x.act.name for x in self.layers], "Activation functions incorrect")

    def test_calculate(self):
        self.layers[-1].calculate(self.data)
        for i in range(len(self.layers)):
            with self.subTest(layer_index=i):
                self.assert_array_equal(self.exp_logits[i], self.layers[i].logits)

    def test_get_weights(self):
        self.assert_array_equal(self.exp_weights, self.layers[-1].get_weights())

    def test_get_grad(self):
        if self.exp_grad is None:
            if any([layer.weights for layer in self.layers]):
                self.skipTest("Expected gradient missing, skipping gradient test.")
            return
        self.assert_array_equal(self.exp_grad, self.layers[-1].get_grad(self.data, self.labels))

    def test_update_weights(self):
        if self.exp_new_weights is None or self.new_weights is None:
            if any([layer.weights for layer in self.layers]):
                self.skipTest("Update or new weights missing, skipping update test.")
            return
        self.layers[-1].update_weights(self.new_weights)
        for i in range(len(self.layers)):
            with self.subTest(layer_index=i):
                self.assert_array_equal(self.exp_new_weights[i], self.layers[i].weights)
        self.layers[-1].update_weights(self.exp_weights)

    def assert_array_equal(self, expected, actual):
        self.assertTrue(np.array_equal(expected, actual), msg=f"Expected array:\n{expected}\nActual array:\n{actual}")


class DenseLayerTest(LayerTest):

    def setUp(self):
        bottom = Layer(4)
        mid = DenseLayer(3, Activation.get("linear"), bottom)
        mid.weights = np.array([[1, 0, 1, 0, 1],
                                [0, 1, 0, 1, 0],
                                [1, 0, 1, 0, 1]])
        top = DenseLayer(2, Activation.get("linear"), mid)
        top.weights = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1]])
        data = np.array([[1, 1],
                         [1, 2],
                         [1, 3],
                         [1, 4]])
        super().set_values(
            layers=[bottom, mid, top],
            data=data,
            labels=np.array([[3, 5],
                             [5, 13]]),
            sizes=[(4,), (3,), (2,)],
            weight_sizes=[None, (3, 5), (2, 4)],
            act_names=[None, "linear", "linear"],
            logits=[data,
                    np.array([[3, 7],
                              [2, 4],
                              [3, 7]]),
                    np.array([[3, 5],
                              [6, 14]])],
            weights=np.array([1, 0, 1, 0,
                              0, 1, 0, 1,
                              1, 0, 1, 0, 1,
                              0, 1, 0, 1, 0,
                              1, 0, 1, 0, 1]),
            grad=np.array([0, 0, 0, 0,
                           2, 10, 6, 10,
                           2, 2, 3, 4, 5,
                           0, 0, 0, 0, 0,
                           2, 2, 3, 4, 5]),
            update=np.array([1, 1, 1, 1,
                             1, 1, 1, 1,
                             1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1]),
            new_weights=[None,
                         np.array([[1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1]]),
                         np.array([[1, 1, 1, 1],
                                   [1, 1, 1, 1]])]
        )


class ConvLayerTest(LayerTest):

    def setUp(self):
        bottom = Layer((4, 4, 1))
        mid = ConvLayer((3, 3, 1), Activation.get("linear"), bottom, stride=2, padding=((1, 1), (1, 1)))
        mid.weights = np.reshape(np.array([[2, 1], [3, 0]]), (2, 2, 1, 1))
        top = ConvLayer((2, 2, 1), Activation.get("linear"), mid)
        top.weights = np.reshape(np.array([[4, -4], [3, -6]]), (2, 2, 1, 1))
        data = np.reshape(np.array([[2, 0, 0, 1],
                                    [1, 0, 1, 0],
                                    [1, 1, 0, 2],
                                    [1, 0, 1, 1]]), (4, 4, 1, 1))
        super().set_values(
            layers=[bottom, mid, top],
            data=data,
            labels=np.reshape(np.array([[-18, -30],
                                        [-10, -12]]), (2, 2, 1, 1)),
            sizes=[(4, 4, 1), (3, 3, 1), (2, 2, 1)],
            weight_sizes=[None, (2, 2, 1, 1), (2, 2, 1, 1)],
            act_names=[None, "linear", "linear"],
            logits=[data,
                    np.reshape(np.array([[0, 0, 3],
                                         [1, 4, 6],
                                         [1, 1, 2]]), (3, 3, 1, 1)),
                    np.reshape(np.array([[-21, -36],
                                         [-15, -17]]), (2, 2, 1, 1))],
            weights=np.array([4, -4,
                              3, -6,
                              2, 1,
                              3, 0]),
            grad=np.array([-25, -68,
                           -37, -63,
                           30, -29,
                           136, -53]),
            update=np.array([1, 1, 1, 1, 0, 1, 0, 1]),
            new_weights=[None,
                         np.reshape(np.array([[0, 1],
                                              [0, 1]]), (2, 2, 1, 1)),
                         np.reshape(np.array([[1, 1],
                                              [1, 1]]), (2, 2, 1, 1))]
        )


class PoolingLayerTest(LayerTest):

    def setUp(self):
        bottom = Layer((4, 4, 3))
        mid = ConvLayer((3, 3, 3), Activation.get("linear"), bottom, stride=2, padding=((1, 1), (1, 1)))
        mid.weights = np.zeros((2, 2, 3, 3))
        mid.weights[:, :, 0, 0] = np.array([[1, 0], [0, 0]])
        mid.weights[:, :, 1, 1] = np.array([[1, 1], [-1, 2]])
        mid.weights[:, :, 2, 2] = np.array([[-4, 1], [5, -7]])
        top = PoolingLayer((2, 2, 3), "max", mid)
        data = np.reshape(np.array([[[1, 1, 1], [1, 1, -1], [1, -5, -3], [1, -4, -8]],
                                    [[1, -1, 1], [1, -3, -1], [1, -1, -3], [1, -4, -8]],
                                    [[1, 0, -2], [1, 1, 0], [1, -1, 0], [1, 0, 1]],
                                    [[1, 1, 2], [1, 0, 0], [1, 1, -2], [1, 0, -9]]]), (4, 4, 3, 1))
        update = np.array([1, -6, 2, 7, -1, 6, -3, 7, 3,
                           1, 7, 0, 3, 0, -3, 0, -5, 8,
                           0, -5, 3, -2, 6, 7, 2, 8, 5,
                           0, 0, 0, -1, -3, -2, 5, 4, 2])
        super().set_values(
            layers=[bottom, mid, top],
            data=data,
            labels=np.reshape(np.array([[[2, 2, 16], [1, 2, 37]],
                                        [[1, 1, 14], [1, 1, 36]]]), (2, 2, 3, 1)),
            sizes=[(4, 4, 3), (3, 3, 3), (2, 2, 3)],
            weight_sizes=[None, (2, 2, 3, 3), None],
            act_names=[None, "linear", "max"],
            logits=[data,
                    np.reshape(np.array([[[0, 2, -7], [0, -11, 16], [0, 4, -40]],
                                         [[0, -1, 15], [1, -7, 1], [1, -4, 37]],
                                         [[0, 1, 2], [1, 1, -2], [1, 0, 36]]]), (3, 3, 3, 1)),
                    np.reshape(np.array([[[1, 2, 16], [1, 4, 37]],
                                         [[1, 1, 15], [1, 1, 37]]]), (2, 2, 3, 1))],
            weights=np.array([1, 0, 0, 0, 1, 0, 0, 0, -4,
                              0, 0, 0, 0, 1, 0, 0, 0, 1,
                              0, 0, 0, 0, -1, 0, 0, 0, 5,
                              0, 0, 0, 0, 2, 0, 0, 0, -7]),
            grad=np.array([-1, 0, 1, 3, 0, -4, 1, 0, -8,
                           -1, 0, 1, 1, 0, -1, 3, 0, 1,
                           -1, 2, 1, -1, -8, 0, 0, -16, 1,
                           -1, 0, 1, 1, 0, 0, 0, 0, -2]),
            update=update,
            new_weights=[None, np.reshape(update, (2, 2, 3, 3)), None]
        )

