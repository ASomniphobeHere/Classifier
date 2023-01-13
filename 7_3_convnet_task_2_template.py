import torch
import torch.nn as nn
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_LEN = 10000
INPUT_SIZE = 28
DEVICE = 'cuda:0'
EPOCHS = 100

class DatasetFashionMNIST(torch.utils.data.Dataset):
    def __init__(self, is_train):
        super().__init__()
        self.data = torchvision.datasets.FashionMNIST(
            root='../data',
            train=is_train,
            download=True
        )

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.data)

    def __getitem__(self, idx):
        # list tuple np.array torch.FloatTensor
        pil_x, y_idx = self.data[idx]
        np_x = np.array(pil_x)
        np_x = np_x / 255
        np_x = np.expand_dims(np_x, axis=0)

        x = torch.FloatTensor(np_x)

        np_y = np.zeros((10,))
        np_y[y_idx] = 1.0

        y = torch.FloatTensor(np_y)
        return x, y


data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetFashionMNIST(is_train=True),
    batch_size=BATCH_SIZE,
    shuffle=True
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=DatasetFashionMNIST(is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False
)

def out_size(in_size, kern, str, pad):
    return int((in_size-kern+2*pad)/str)+1

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        end_channels = 16
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(in_channels=5, out_channels=9, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(in_channels=9, out_channels=end_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.25)
        )
        firstConvPool = out_size(in_size=out_size(in_size=INPUT_SIZE, kern=3, str=1, pad=1), kern=2, str=2, pad=1)
        secondConvPool = out_size(in_size=out_size(in_size=firstConvPool, kern=3, str=1, pad=1), kern=2, str=2, pad=1)
        thirdConvPool = out_size(in_size=secondConvPool, kern=3, str=1, pad=1)
        end_size = thirdConvPool
        # print(end_size)
        self.fc = nn.Linear(
            in_features=end_channels*end_size*end_size,
            out_features=10,
            device=DEVICE
        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.encoder.forward(x)
        # print(out.size())
        out_flat = out.view(batch_size, -1)
        logits = self.fc.forward(out_flat)
        y_prim = torch.softmax(logits, dim=-1)
        return y_prim


model = Model()
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, EPOCHS+1):
    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y in data_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_prim = model.forward(x)

            loss = torch.mean(-y * torch.log(y_prim + 1e-8))

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_y = y.cpu().data.numpy()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.average((idx_y == idx_y_prim) * 1.0)

            metrics_epoch[f'{stage}_acc'].append(acc)
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 3)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    if epoch % 10 == 0:
        plts = []
        c = 0
        for key, value in metrics.items():
            plts += plt.plot(value, f'C{c}', label=key)
            ax = plt.twinx()
            c += 1

        plt.legend(plts, [it.get_label() for it in plts])
        plt.show()