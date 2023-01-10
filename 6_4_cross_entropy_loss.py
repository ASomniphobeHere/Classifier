import os
import pickle
import time
import matplotlib
import sys
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 14) # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-1
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.8

class Dataset:
    def __init__(self):
        super().__init__()
        path_dataset = '../data/cardekho_india_dataset.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/cardekho_india_dataset.pkl',
                path_dataset,
                progress=True
            )
        with open(f'{path_dataset}', 'rb') as fp:
            X_tmp, Y_tmp, self.labels = pickle.load(fp)

        # self.labels[1] = ['Diesel', 'Petrol', 'LPG', 'CNG']
        X_tmp = np.array(X_tmp)
        X_classes = np.array(X_tmp[:, :4])

        self.Y = X_classes[:, 1]

        self.Y_prob = np.zeros((len(self.Y), len(self.labels[1])))
        idxes_range = range(len(self.Y))
        self.Y_prob[idxes_range, self.Y] = 1
        print(self.Y_prob)

        self.X_classes = np.concatenate((X_classes[:, :1], X_classes[:, 2:]), axis=-1)
        # TODO convert to one-hot-encoded probs

        X_tmp = np.array(X_tmp[:, 4:]).astype(np.float32)
        Y_tmp = np.squeeze(np.expand_dims(Y_tmp, axis=-1).astype(np.float32), axis=-1)
        self.X = np.concatenate((X_tmp, Y_tmp), axis=-1)
        X_max = np.max(self.X, axis=0) # (7, )
        X_min = np.min(self.X, axis=0)
        self.X = (self.X - (X_max + X_min) * 0.5) / (X_max - X_min) * 0.5

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.array(self.X[idx]), np.array(self.X_classes[idx]), np.array(self.Y_prob[idx])


class DataLoader:
    def __init__(
            self,
            dataset,
            idx_start, idx_end,
            batch_size
    ):
        super().__init__()
        self.dataset = dataset
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.batch_size = batch_size
        self.idx_batch = 0

    def __len__(self):
        return (self.idx_end - self.idx_start - self.batch_size) // self.batch_size

    def __iter__(self):
        self.idx_batch = 0
        return self

    def __next__(self):
        if self.idx_batch > len(self):
            raise StopIteration()
        idx_start = self.idx_batch * self.batch_size + self.idx_start
        idx_end = idx_start + self.batch_size
        batch = self.dataset[idx_start:idx_end]
        self.idx_batch += 1
        return batch

dataset_full = Dataset()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)

dataloader_train = DataLoader(
    dataset_full,
    idx_start=0,
    idx_end=train_test_split,
    batch_size=BATCH_SIZE
)
dataloader_test = DataLoader(
    dataset_full,
    idx_start=train_test_split,
    idx_end=len(dataset_full),
    batch_size=BATCH_SIZE
)

class Variable:
    def __init__(self, value, grad=None):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)
        if grad is not None:
            self.grad = grad


class LayerLinear:
    def __init__(self, in_features: int, out_features: int):
        self.W:  Variable = Variable(
            value=np.random.uniform(low=-1, size=(in_features, out_features)),
            grad=np.zeros(shape=(BATCH_SIZE, in_features, out_features))
        )
        self.b: Variable = Variable(
            value=np.zeros(shape=(out_features,)),
            grad=np.zeros(shape=(BATCH_SIZE, out_features))
        )
        self.x: Variable = None
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(
            np.squeeze(self.W.value.T @ np.expand_dims(x.value, axis=-1), axis=-1) + self.b.value
        )
        return self.output

    def backward(self):
        self.b.grad += 1 * self.output.grad
        self.W.grad += np.expand_dims(self.x.value, axis=-1) @ np.expand_dims(self.output.grad, axis=-2)
        self.x.grad += np.squeeze(self.W.value @ np.expand_dims(self.output.grad, axis=-1), axis=-1)

class LayerSigmoid():
    def __init__(self):
        self.x: Variable = None
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(1.0 / (1.0 + np.exp(-x.value)))
        return self.output

    def backward(self):
        self.x.grad += self.output.value * (1.0 - self.output.value) * self.output.grad

class LayerSoftmax():
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x):
        self.x = x
        self.x.value-= np.max(self.x.value, axis=-1, keepdims=True)
        self.output = Variable(
            np.exp(self.x.value)/np.expand_dims(np.sum(np.exp(self.x.value), axis=-1), axis=-1)
        )
        return self.output

    def backward(self):
        size = self.x.value.shape[-1]
        identity = np.repeat(np.expand_dims(np.identity(size), axis=0), BATCH_SIZE, axis=0) # identity matrix with extra batch dimension
        x_repeat = np.repeat(np.expand_dims(self.output.value, axis=-2), size, axis=-2)
        # print(x_repeat)
        """ [a0 a1 a2 a3]
            [a0 a1 a2 a3]
            [a0 a1 a2 a3]
            [a0 a1 a2 a3]"""
        x_repeat_t = np.repeat(np.expand_dims(self.output.value, axis=-1), size, axis=-1)
        # print(x_repeat_t)
        """ [a0 a0 a0 a0]
            [a1 a1 a1 a1]
            [a2 a2 a2 a2]
            [a3 a3 a3 a3]"""
        J=(identity-x_repeat_t)*x_repeat
        """ [a0(1-a0) a1(0-a0) a2(0-a0) a3(0-a0)]
            [a0(0-a1) a1(1-a1) a2(0-a1) a3(0-a1)]
            [a0(0-a2) a1(0-a2) a2(1-a2) a3(0-a2)]
            [a0(0-a3) a1(0-a3) a2(0-a3) a3(1-a3)]"""
        a = self.output.value
        self.x.grad += np.squeeze(J @ np.expand_dims(self.output.grad, axis=-1), axis=-1)

class LossCrossEntropy():
    def __init__(self):
        self.y_prim = None

    def forward(self, y, y_prim):
        self.y = y
        self.y_prim = y_prim
        return np.mean(-self.y.value * np.log(self.y_prim.value + 1e-8))

    def backward(self):
         self.y_prim.grad = -self.y.value/(self.y_prim.value + 1e-8)


class LayerEmbedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.x: Variable = None
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.emb_matrix = Variable(np.random.random((num_embeddings, embedding_dim)) - 0.5)
        self.output: Variable = None

    def forward(self, x: Variable):
        self.x = x
        #TODO

    def backward(self):
         self.emb_matrix.grad += 0 #TODO


class Model:
    def __init__(self):
        self.layers = [
            LayerLinear(in_features=4, out_features=8),
            LayerSigmoid(),
            LayerLinear(in_features=8, out_features=8),
            LayerSigmoid(),
            LayerLinear(in_features=8, out_features=4),
            LayerSoftmax()
        ]

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self):
        variables = []
        for layer in self.layers:
            if type(layer) == LayerLinear:
                variables.append(layer.W)
                variables.append(layer.b)
        return variables


class OptimizerSGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            # W'= W - dW * alpha
            param.value -= np.mean(param.grad, axis=0) * self.learning_rate

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)


model = Model()
optimizer = OptimizerSGD(
    model.parameters(),
    learning_rate=LEARNING_RATE
)
loss_fn = LossCrossEntropy()


loss_plot_train = []
loss_plot_test = []
acc_plot_train = []
acc_plot_test = []

for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        nrmse = []
        accs = []
        for x, x_classes, y in dataloader:

            y_prim = model.forward(Variable(value=x))
            loss = loss_fn.forward(Variable(value=y), y_prim)
            y_idx = np.argmax(y, axis=-1)
            y_prim_idx = np.argmax(y_prim.value, axis=-1)
            acc = np.mean((y_idx == y_prim_idx) * 1.0)
            losses.append(loss)
            accs.append(acc)

            if dataloader == dataloader_train:
                loss_fn.backward()
                model.backward()

                optimizer.step()
                optimizer.zero_grad()

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
            acc_plot_train.append(np.mean(accs))
        else:
            loss_plot_test.append(np.mean(losses))
            acc_plot_test.append(np.mean(accs))

    print(
        f'epoch: {epoch} '
        f'loss_train: {loss_plot_train[-1]} '
        f'loss_test: {loss_plot_test[-1]}'
        f'acc_train: {acc_plot_train[-1]} '
        f'acc_test: {acc_plot_test[-1]}'
    )

    if epoch % 1000 == 0:
        _, axes = plt.subplots(nrows=2, ncols=1)
        ax1 = axes[0]
        ax1.set_title("Loss")
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        ax1 = axes[1]
        ax1.set_title("Acc")
        ax1.plot(acc_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(acc_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Acc.")
        plt.show()