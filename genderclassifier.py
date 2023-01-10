import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from pydub import AudioSegment
from pydub.silence import split_on_silence
import time
from librosa.feature import mfcc
import torch
import torch.nn as nn

BATCH_SIZE = 64
N_MFCC=40
TRAIN_TEST_SPLIT=0.7
LEARNING_RATE=0.01
EPOCHS=100

# for samplem4a in tqdm(os.listdir(r"wav/males")):
#     temp = AudioSegment.from_file("m4a/males/" + samplem4a, format="m4a")
#     temp.export(("wav/males/" + samplem4a.split('.')[0] + ".wav"), format="wav")

def remove_silence(audiosample):
    audiochunks = split_on_silence(
        audiosample,
        min_silence_len=100,
        silence_thresh=-45,
        keep_silence=50
    )
    result = AudioSegment.empty()
    for chunk in audiochunks:
        result+=chunk
    return result

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

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=N_MFCC, out_features=16),
            nn.Sigmoid(),
            nn.Linear(in_features=16, out_features=2)
        )

    def forward(self, x):
        out = self.encoder.forward(x)
        y_prim = torch.softmax(out, dim=-1)
        return y_prim

# for temp2 in tqdm(os.listdir(r"wav/females")):#[:50]):
#     audio = remove_silence(AudioSegment.from_wav("wav/females/" + temp2))
#     audio.export("wavfinal/f_"+temp2, "wav")

# for temp2 in tqdm(os.listdir(r"wav/males")):#[:50]):
#     audio = remove_silence(AudioSegment.from_wav("wav/males/" + temp2))
#     audio.export("wavfinal/m_"+temp2, "wav")

male_voices = []
female_voices = []

for audio in tqdm(os.listdir("wavfinal")):#[4150:5100]):
    # print(audio)
    if audio.split("_")[0] == "f" and len(female_voices)<500:
        female_voices.append(librosa.load("wavfinal/" + audio))
    elif audio.split("_")[0] == "m" and len(male_voices)<500:
        male_voices.append(librosa.load("wavfinal/" + audio))

male_features = np.zeros(shape=(1, N_MFCC))
female_features = np.zeros(shape=(1, N_MFCC))
for temp3 in tqdm(female_voices):
    female_features = np.vstack((female_features, np.transpose(mfcc(y=temp3[0], sr=temp3[1], n_mfcc=N_MFCC)[:100], (1, 0))))
for temp3 in tqdm(male_voices):
    male_features = np.vstack((male_features, np.transpose(mfcc(y=temp3[0], sr=temp3[1], n_mfcc=N_MFCC)[:100], (1, 0))))
female_features = female_features[1:]
male_features = male_features[1:]
X = np.vstack((male_features, female_features))
X = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))
print(X)
Y = np.append([0] * len(male_features), [1] * len(female_features))
print(X.shape, Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7)

clf = SVC(kernel='rbf')
print("fitting")
timesince = time.time()
clf.fit(X_train[:50000], Y_train[:50000])
print("time in seconds:", time.time() - timesince)
print()
print("validating")
timesince = time.time()
print("score: ", clf.score(X_train[:50000], Y_train[:50000]))
print("time in seconds: ", time.time() - timesince)
print()
print("testing")
timesince = time.time()
print("score:", clf.score(X_test[:50000], Y_test[:50000]))
print("time in sconds: ", time.time() - timesince)

predicted = clf.predict(X_test[:50000])

result = metrics.ConfusionMatrixDisplay(confusion_matrix = metrics.confusion_matrix(Y_test[:50000], predicted), display_labels = ["Male", "Female"])
result.plot()
plt.show()

class Dataset:
    def __init__(self):
        super().__init__()
        self.Y = Y

        self.Y_prob = np.zeros((len(self.Y), 2))
        idxes_range = range(len(self.Y))
        self.Y_prob[idxes_range, self.Y] = 1
        print(self.Y_prob)

        self.X = X
        X_max = np.max(self.X, axis=0) # (7, )
        X_min = np.min(self.X, axis=0)
        self.X = (self.X - X_min)/(X_max - X_min)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.array(self.X[idx]), np.array(self.Y_prob[idx])

dataset_full = Dataset()

dataset_train = DataLoader(
    dataset_full,
    idx_start=0,
    idx_end=int(TRAIN_TEST_SPLIT*len(dataset_full)),
    batch_size=BATCH_SIZE
 )
dataset_test = DataLoader(
    dataset_full,
    idx_start=int(TRAIN_TEST_SPLIT*len(dataset_full)),
    idx_end=len(dataset_full),
    batch_size=BATCH_SIZE
 )

model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, EPOCHS+1):
    for data_loader in [dataset_train, dataset_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == dataset_test:
            stage = 'test'

        for x, y in data_loader:

            y_prim = model.forward(x)

            loss = torch.mean(-y * torch.log(y_prim + 1e-8))

            if data_loader == dataset_train:
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
                metrics_strs.append(f'{key}: {round(value, 2)}')

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