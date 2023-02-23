import os
import pickle
import time
import matplotlib
import sys
import torch
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F
from time import sleep

plt.rcParams["figure.figsize"] = (12, 7) # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-3
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.7


class Dataset(torch.utils.data.Dataset):
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
            X, self.Y, self.labels = pickle.load(fp)

        X = np.array(X)
        self.X_classes = np.array(X[:, :4])

        self.X = np.array(X[:, 4:]).astype(np.float32)
        X_max = np.max(self.X, axis=0) # (7, )
        X_min = np.min(self.X, axis=0)
        self.X = (self.X - (X_max + X_min) * 0.5) / (X_max - X_min) * 0.5
        print(self.X.shape)

        self.Y = np.array(self.Y).astype(np.float32)
        self.Y = self.Y[:, 1]
        Y_max  = np.max(self.Y)
        Y_min  = np.min(self.Y)
        self.Y = (self.Y - (Y_max + Y_min) * 0.5) / (Y_max - Y_min) * 0.5
        print(self.Y.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.array(self.X[idx]), np.array(self.X_classes[idx]), np.expand_dims(self.Y[idx], axis=-1)

dataset_full = Dataset()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

dataloader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=2 + 4*3, out_features=40),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=40, out_features=40),
            torch.nn.Sigmoid(),
            torch.nn.Linear(in_features=40, out_features=1)
        )

        self.embs = torch.nn.ModuleList()
        for i in range(4):
            self.embs.append(
                torch.nn.Embedding(
                    num_embeddings=len(dataset_full.labels[i]),
                    embedding_dim=3
                )
            )

    def forward(self, x, x_classes):

        x_emb_list = []
        for i, emb in enumerate(self.embs):
            x_emb_list.append(
                emb.forward(x_classes[:, i])
            )
        x_emb = torch.cat(x_emb_list, dim=-1)
        x_cat = torch.cat([x, x_emb], dim=-1)

        y_prim = self.layers.forward(x_cat)
        return y_prim

class LossHuber(torch.nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, y_prim, y):
        return torch.mean(self.delta**2 * (torch.sqrt(1+((y-y_prim)/self.delta)**2)-1))


model = Model()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
loss_fn = LossHuber(delta=0.5)

loss_plot_train = []
loss_plot_test = []
for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        for x, x_classes, y in dataloader:
            y_prim = model.forward(x, x_classes)
            # print(y_prim, y)
            loss = loss_fn.forward(y_prim, y)

            losses.append(loss.item())

            if dataloader == dataloader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
        else:
            loss_plot_test.append(np.mean(losses))

    print(f'epoch: {epoch} loss_train: {loss_plot_train[-1]} loss_test: {loss_plot_test[-1]}')

    if epoch % 50 == 0:
        fig, ax1 = plt.subplots()
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        plt.show()