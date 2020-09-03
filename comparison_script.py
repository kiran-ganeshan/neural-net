import numpy as np
import time

np.set_printoptions(edgeitems=10, linewidth=200)
import matplotlib.pyplot as plt
import itertools
from model import *
from optimizer import *

num_nets = 3
epochs = 5
toolbar_width = 30
alpha = 0.01
lamb = 0.03
dampening = 0.1
alpha_max = 1
alpha_decay = 2
max_decay = 50
vectors = 4
min_dec = 0.0001
min_curv = 0.9

act_funcs = ["sigmoid", "relu", "tanh", "elu", "lrelu"]
optimizers = ["gd", "mom", "rmsprop", "adam"]
act_names = {"sigmoid": "Sigmoid", "relu": "ReLU", "tanh": "Tanh", "elu": "ELU", "lrelu": "LReLU"}
opt_names = {'gd': 'Gradient Descent', 'mom': 'Momentum', 'rmsprop': 'RMSProp', 'adam': 'Adam', 'l-bfgs': 'L-BFGS'}

def setup_kwargs(optimizer):
    kwargs = {"lamb": 0.03, "batch_size": 60, "alpha": alpha}
    if optimizer in ["mom", "adam"]:
        kwargs["grad_decay"] = dampening
    if optimizer in ["rmsprop", "adam"]:
        kwargs["square_decay"] = dampening
    if optimizer == "l-bfgs":
        kwargs["batch_size"] = 60000
        kwargs["vectors"] = vectors
        kwargs["alpha_max"] = alpha_max
        kwargs["alpha_decay"] = alpha_decay
        kwargs["max_decay"] = max_decay
        kwargs["min_dec"] = min_dec
        kwargs["min_curv"] = min_curv
    return kwargs

def print_attribute_table(attribute):
    act_funcs = ["sigmoid", "relu", "tanh", "elu", "lrelu"]
    act_names = {"sigmoid": "Sigmoid", "relu": "ReLU", "tanh": "Tanh", "elu": "ELU", "lrelu": "LReLU"}
    optimizers = ["gd", "mom", "rmsprop", "adam"]
    opt_names = {'gd': 'Gradient Descent', 'mom': 'Momentum', 'rmsprop': 'RMSProp', 'adam': 'Adam', 'l-bfgs': 'L-BFGS'}
    act_func_names = map(lambda x: act_names[x], act_funcs)
    row_format = "{:>15}" * 6
    print(row_format.format("", *act_func_names))
    for optimizer in optimizers:
        results = map(lambda x: str(round(attribute[x][optimizer], 3)), act_funcs)
        print(row_format.format(opt_names[optimizer], *results))

# Read MNIST Data from files
print("Reading MNIST training data...")
X = np.transpose(np.load('train_imgs.npy', allow_pickle=True))
y = np.transpose(np.load('train_labels.npy', allow_pickle=True))
print("Reading MNIST test data...")
Xtest = 1/255.0 * np.transpose(np.load('test_imgs.npy', allow_pickle=True))
ytest = np.transpose(np.load('test_labels.npy', allow_pickle=True))

# Instantiate Model and Optimizer
model, runtime, cost, accuracy = dict(), dict(), dict(), dict()
for act_func in act_funcs:
    model[act_func], runtime[act_func], cost[act_func], accuracy[act_func] = dict(), dict(), dict(), dict()
for act_func, optimizer in itertools.product(act_funcs, optimizers):
    model[act_func][optimizer] = (NeuralNetwork((784, 300, 10), act_func, f'weights/{act_func}{optimizer}'),) * num_nets
    print("Training " + act_names[act_func] + " with " + opt_names[optimizer] + ":")
    sample_runtime, sample_cost, sample_accuracy = list(), list(), list()
    for network in model[act_func][optimizer]:
        opt = Optimizer.get(optimizer, network, X, y)
        opt.compile(**setup_kwargs(optimizer))
        start_time = time.time()
        opt.train(epochs)
        sample_runtime.append(time.time() - start_time)
        sample_cost.append(network.get_cost(Xtest, ytest, lamb))
        sample_accuracy.append(network.get_accuracy(Xtest, ytest))
        print('')
    sample_max = sample_accuracy.index(max(sample_accuracy))
    runtime[act_func][optimizer] = sample_runtime[sample_max]
    cost[act_func][optimizer] = sample_cost[sample_max]
    accuracy[act_func][optimizer] = sample_accuracy[sample_max]
print("\nRuntime:")
print_attribute_table(runtime)
print("\nCost:")
print_attribute_table(cost)
print("\nAccuracy:")
print_attribute_table(accuracy)

