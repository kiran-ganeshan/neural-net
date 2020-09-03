import sys
from math import floor
from model import *
from activation import *
import time


def write_progress_bar(iter, total):
    progress_bar = '=' * ((30 * (iter + 1)) // total) + '>'
    spaces = ' ' * (30 - (30 * (iter + 1)) // total)
    percentage = round(100 * (iter + 1) / total, 1)
    sys.stdout.write('\r')
    sys.stdout.write(f"[{progress_bar}{spaces}] {percentage}%")
    sys.stdout.flush()

class Optimizer:

    def __init__(self, model, data, labels, plot_cost=False):
        self.num_layers = len(model.layer_sizes)
        self.layer_sizes = model.layer_sizes
        self.model = model
        self.data = data
        self.labels = labels
        self.requires_cost = False
        self.plot_cost = plot_cost
        if plot_cost:
            self.cost = list()

    def compile(self, **kwargs):
        self.lamb = kwargs['lamb']
        self.batch_size = kwargs['batch_size']

    def train(self, epoch):
        count = 0
        iter_per_epoch = floor(self.data.shape[-1] / self.batch_size)
        total = epoch * iter_per_epoch
        write_progress_bar(-1, total)
        indices = [k * self.batch_size for k in range(1, iter_per_epoch)]
        data_slices, label_slices = np.split(self.data, indices, -1), np.split(self.labels, indices, -1)
        for e in range(epoch):
            for k in range(iter_per_epoch):
                weights = self.model.get_weights()
                diff = self.model.get_grad(data_slices[k], label_slices[k], self.lamb)
                if not self.requires_cost:
                    self.model.update_weights(self.update(weights, diff))
                    if self.plot_cost:
                        self.cost.append(self.model.get_cost(data_slices[k], label_slices[k], self.lamb))
                else:
                    cost = self.model.get_cost(data_slices[k], label_slices[k], self.lamb)
                    self.model.update_weights(self.update(weights, diff, cost))
                    if self.plot_cost:
                        self.cost.append(cost)
                write_progress_bar(count, total)
                count += 1
        self.model.save_weights()

    def update(self, weights, diff, cost=None):
        pass

    @staticmethod
    def get(opt, model, data, labels):
        if opt == "gd":
            return GDOptimizer(model, data, labels)
        elif opt == "mom":
            return MomentumOptimizer(model, data, labels)
        elif opt == "rmsprop":
            return RMSPropOptimizer(model, data, labels)
        elif opt == "adam":
            return AdamOptimizer(model, data, labels)
        elif opt == "l-bfgs":
            return LBFGSOptimizer(model, data, labels)
        else:
            return Optimizer(model, data, labels)


class GDOptimizer(Optimizer):

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.name = "GD"
        self.alpha = kwargs["alpha"]

    def update(self, weights, diff, cost=None):
        return weights - self.alpha * diff


class MomentumOptimizer(GDOptimizer):

    def __init__(self, model, data, labels):
        super().__init__(model, data, labels)
        self.velocity = np.zeros(shape=(model.get_weights().size,))

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.name = "Momentum"
        self.damp = kwargs["grad_decay"]

    def update(self, weights, diff, cost=None):
        self.velocity *= self.damp
        self.velocity += (1 - self.damp) * diff
        return super().update(weights, self.velocity)


class RMSPropOptimizer(GDOptimizer):

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.name = "RMS-Prop"
        self.decay = kwargs["square_decay"]
        self.squared = 1

    def update(self, weights, diff, cost=None):
        self.squared *= 1 - self.decay
        self.squared += self.decay * np.sum(weights * weights)
        weights -= self.alpha * diff / (np.sqrt(self.squared) + 1e-8)
        return weights


class AdamOptimizer(MomentumOptimizer, RMSPropOptimizer):

    def compile(self, **kwargs):
        MomentumOptimizer.compile(self, **kwargs)
        RMSPropOptimizer.compile(self, **kwargs)
        self.name = "Adam"

    def update(self, weights, diff, cost=None):
        vel = (weights - MomentumOptimizer.update(self, weights, diff)) / self.alpha
        return RMSPropOptimizer.update(self, weights, vel)


class LBFGSOptimizer(Optimizer):

    def __init__(self, model, data, labels):
        super().__init__(model, data, labels)
        self.s = list()
        self.y = list()
        self.rho = list()
        self.filling = True
        self.requires_cost = True

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.vectors = kwargs["vectors"]
        self.alpha_max = kwargs["alpha_max"]
        self.alpha_decay = kwargs["alpha_decay"]
        self.max_decay = kwargs["max_decay"]
        self.min_dec = kwargs["min_dec"]
        self.min_curv = kwargs["min_curv"]
        self.weights = self.model.get_weights()

    def update(self, weights, diff, cost=None):
        dir = self.__get_dir(diff)
        return self.__line_search(dir, diff, cost)

    def __line_search(self, dir, diff, cost):
        alpha = self.alpha_max
        weights = self.model.get_weights()
        for k in range(0, self.max_decay):
            new_weights = weights + alpha * dir
            self.model.update_weights(new_weights)
            new_cost = self.model.get_cost(self.data, self.labels, self.lamb)
            dot_product = np.sum(dir * diff)
            print(f'\nRate of increase: {round(dot_product, 3)}')
            print(f'Cost: {round(cost, 3)} -> {round(new_cost, 3)}')
            if new_cost <= cost + self.min_dec * alpha * dot_product:
                new_diff = self.model.get_grad(self.data, self.labels, self.lamb)
                print(f'Cost Derivative: {-round(dot_product, 3)} -> {-round(np.sum(dir * new_diff), 3)}')
                if -np.sum(dir * new_diff) <= -self.min_curv * dot_product:
                    print(new_weights)
                    self.s.append(new_weights - weights)
                    self.y.append(new_diff - diff)
                    self.rho.append(np.sum(self.s[-1] * self.y[-1]))
                    if len(self.s) > self.vectors:
                        map(lambda x: x.pop(0), [self.s, self.y, self.rho])
                    return new_weights
            alpha = alpha / self.alpha_decay
        print('Line search failed. Reducing requirements.')
        self.min_dec = self.min_dec / 2.0
        return self.__line_search(dir)

    def __get_dir(self, diff):
        alpha = list()
        length = min(self.vectors, len(self.s))
        for i in reversed(range(length)):
            alpha.insert(0, np.sum(self.s[i] * diff) / self.rho[i])
            diff -= alpha[0] * self.y[i]
        if length > 0:
            diff *= self.rho[-1] / np.sum(self.y[-1] * self.y[-1])
        for i in range(length):
            beta = np.sum(self.y[i] * self.rho[i]) / self.rho[i]
            diff += (alpha[i] - beta) * self.s[i]
        return -diff
