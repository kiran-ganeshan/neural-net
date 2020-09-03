import numpy as np
import time

np.set_printoptions(edgeitems=10, linewidth=200)
import matplotlib.pyplot as plt
from model import *
from optimizer import *
from tabulate import tabulate
epochs = 10
toolbar_width = 30
alpha = 0.05
lamb = 0.03
dampening = 0.1
alpha_max = 1
alpha_decay = 2
max_decay = 50
vectors = 4
min_dec = 0.0001
min_curv = 0.9
batch_size = 600
grad_decay = 0.2

# Read MNIST Data from files
print("Reading MNIST training data...")
X = np.reshape(np.transpose(np.load('train_imgs.npy', allow_pickle=True)), (28, 28, 1, 60000))
y = np.transpose(np.load('train_labels.npy', allow_pickle=True))
print("Reading MNIST test data...")
Xtest = np.reshape(1/255.0 * np.transpose(np.load('test_imgs.npy', allow_pickle=True)), (28, 28, 1, 10000))
ytest = np.transpose(np.load('test_labels.npy', allow_pickle=True))

X = np.reshape(X, (784, 60000))
Xtest = np.reshape(Xtest, (784, 10000))

model = ConvNeuralNetwork(("input", "conv", "pool", "conv", "dense", "dense", "dense"),
                          (784, (28, 28, 1), (26, 26, 1), (20, 20, 1), 300, 100, 10), "tanh", "max", "throwaway")
optimizer = Optimizer.get("gd", model, X, y)
optimizer.compile(batch_size=batch_size, alpha=alpha, lamb=lamb, grad_decay=grad_decay)

# Instantiate core metrics
print("Training:")
train_time = 0

train_costs, test_costs, train_acc, test_acc = list(), list(), list(), list()

# Train network
train_time = time.time() - train_time
optimizer.train(epochs)
train_time = time.time() - train_time

# Calculate core metrics
train_costs.append(model.get_cost(X, y, lamb))
test_costs.append(model.get_cost(Xtest, ytest, lamb))
train_acc.append(model.get_accuracy(X, y))
test_acc.append(model.get_accuracy(Xtest, ytest))

# Print and save results
print('')
print("Completed training.")
print(f'Training Costs: {[round(x, 3) for x in train_costs]}')
print(f'Training Accuracies: {[round(x, 3) for x in train_acc]}')
print(f'Testing Costs: {[round(x, 3) for x in test_costs]}')
print(f'Testing Accuracies: {[round(x, 3) for x in test_acc]}')
print(f'Train time: {round(train_time, 3)}')
# create figure
# fig = plt.figure()

# add loss plot
# ax1 = fig.add_subplot(1,2,1)
# ax1.plot(range(epochs+1), train_costs, label='Training Set Loss')
# ax1.plot(range(epochs+1), test_costs, label='Test Set Loss')
# ax1.set_ylabel('Loss')
# ax1.set_xlabel('Training Iteration')
# ax1.set_title('Loss')
# ax1.legend()

# add accuracy plot
# ax2 = fig.add_subplot(1,2,2)
# ax2.plot(range(epochs+1), train_acc, label='Training Set Accuracy')
# ax2.plot(range(epochs+1), test_acc, label='Test Set Accuracy')
# ax2.set_ylabel('Accuracy')
# ax2.set_xlabel('Training Iteration')
# ax2.set_title('Accuracy')
# ax2.legend()

# show figure
#plt.show()


