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
import torch.utils.data

BATCH_SIZE = 64
N_MFCC = 30# mfcc extracted
TRAIN_TEST_SPLIT = 0.7
LEARNING_RATE = 0.005
EPOCHS = 1000
LIM_SIZE = 1000#limit sample size for each gender
MFCC_LIM_SIZE = 100#make sure mfcc used in a sample divides this

torch.set_default_dtype(torch.float32)

def remove_silence(audiosample): #returns audio sample with each silence reduced to 100 ms
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

class Dataset:
    def __init__(self, X, Y):
        super().__init__()

        self.Y = Y
        self.Y_prob = np.zeros((len(self.Y), 2))
        idxes_range = range(len(self.Y))
        self.Y_prob[idxes_range, self.Y] = 1
        self.Y_prob = torch.Tensor(self.Y_prob)

        self.X = X
        self.X = torch.Tensor(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y_prob[idx]

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=3*N_MFCC, out_features=128),
            nn.LeakyReLU(-0.2),
            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(-0.2),
            nn.Linear(in_features=64, out_features=2)
        )

    def forward(self, x):
        out = self.encoder.forward(x)
        y_prim = torch.softmax(out, dim=-1)
        return y_prim


# #export to wav and combine to one folder with labels, DO ONLY ONCE
# for temp2 in tqdm(os.listdir(r"wav/females")):#[:50]):
#     audio = remove_silence(AudioSegment.from_wav("wav/females/" + temp2))
#     audio.export("wavfinal/f_"+temp2, "wav")
# prepare starting dataset
# for temp2 in tqdm(os.listdir(r"wav/males")):#[:50]):
#     audio = remove_silence(AudioSegment.from_wav("wav/males/" + temp2))
#     audio.export("wavfinal/m_"+temp2, "wav")

male_voices = []
female_voices = []

#load audio in lists
for audio in tqdm(os.listdir("wavfinal")):#[4150:5100]): #
    # print(audio)
    if audio.split("_")[0] == "f" and len(female_voices)<LIM_SIZE:# second half for testing and shorter runtime
        female_voices.append(librosa.load("wavfinal/" + audio))
    elif audio.split("_")[0] == "m" and len(male_voices)<LIM_SIZE:
        male_voices.append(librosa.load("wavfinal/" + audio))

male_features = np.zeros(shape=(3*N_MFCC, 1)) # initialize mfcc arrays
female_features = np.zeros(shape=(3*N_MFCC, 1))

#extract mfcc
for female_voice in tqdm(female_voices):
    content, sr = female_voice
    extract_mfcc = mfcc(y=content, sr=sr, n_mfcc=N_MFCC)[:, :MFCC_LIM_SIZE]#so that SVC doesnt take forever to fit
    if extract_mfcc.shape[1] < MFCC_LIM_SIZE:
        continue
    delta1 = librosa.feature.delta(extract_mfcc)
    delta2 = librosa.feature.delta(extract_mfcc, order=2)
    extract_mfcc = np.vstack((extract_mfcc, delta1, delta2))
    female_features = np.hstack((female_features, extract_mfcc))#add the mfcc to feature arrays row-wise, combine mfcc in one big list
for male_voice in tqdm(male_voices):
    content, sr = male_voice
    extract_mfcc = mfcc(y=content, sr=sr, n_mfcc=N_MFCC)[:, :MFCC_LIM_SIZE]
    if extract_mfcc.shape[1] < MFCC_LIM_SIZE:
        continue
    delta1 = librosa.feature.delta(extract_mfcc)
    delta2 = librosa.feature.delta(extract_mfcc, order=2)
    extract_mfcc = np.vstack((extract_mfcc, delta1, delta2))
    male_features = np.hstack((male_features, extract_mfcc))

female_features = female_features.transpose()
male_features = male_features.transpose()
female_features = female_features[1:]#remove first vector(empty)
male_features = male_features[1:]
# female_features = female_features.reshape(-1, 3*3*N_MFCC)# n*3*N_MFCC, n mfcc used in a sample
# male_features = male_features.reshape(-1, 3*3*N_MFCC)
print(male_features.shape)

X = np.vstack((male_features, female_features))#combine both genders in one dataset
X = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))# normalize 0-1
# print(X)
Y = np.append([0] * len(male_features), [1] * len(female_features))#expected output array
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7)

#SVM approach
#SVC uses mfcc from individual windows, not together in a sample

clf = SVC(kernel='rbf')
print("fitting")
timesince = time.time()
clf.fit(X_train[:10000], Y_train[:10000])
print("time in seconds:", time.time() - timesince, "\n")

print("validating")
timesince = time.time()
print("score: ", clf.score(X_train[:10000], Y_train[:10000]))
print("time in seconds: ", time.time() - timesince, "\n")

print("testing")
timesince = time.time()
print("score:", clf.score(X_test[:5000], Y_test[:5000]))
print("time in seconds: ", time.time() - timesince, "\n")

#confusion matrix and metrics

predicted = clf.predict(X_test[:5000])
c_d = metrics.confusion_matrix(Y_test[:5000], predicted)
c_d = c_d/1.0

accuracy = (c_d[0, 0] + c_d[1, 1])/np.sum(c_d)
precision = c_d[0, 0]/(c_d[0, 0] + c_d[0, 1])
recall = c_d[0, 0]/(c_d[0, 0] + c_d[1, 0])
f1 = 2*precision*recall/(precision + recall)
print("male")
print("accuracy:", round(accuracy, 3), "precision:", round(precision, 3), "recall:", round(recall, 3), "f1:", round(f1, 3))

accuracy = (c_d[0, 0] + c_d[1, 1])/np.sum(c_d)
precision = c_d[1, 1]/(c_d[1, 1] + c_d[1, 0])
recall = c_d[1, 1]/(c_d[1, 1] + c_d[0, 1])
f1 = 2*precision*recall/(precision + recall)
print()
print("female")
print("accuracy:", round(accuracy, 3), "precision:", round(precision, 3), "recall:", round(recall, 3), "f1:", round(f1, 3))

print(c_d)
c_d = c_d.astype(int)
# create confusion matrix plot from generated matrix
result = metrics.ConfusionMatrixDisplay(confusion_matrix = c_d, display_labels = ["Male", "Female"])
result.plot()
plt.show()

#DNN approach

dataset_full = Dataset(X, Y)
train_test_split = int(len(dataset_full)*TRAIN_TEST_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full)-train_test_split],
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

model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
metricsdict = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metricsdict[f'{stage}_{metric}'] = []

for epoch in range(1, EPOCHS+1):
    for data_loader in [dataloader_train, dataloader_test]:
        metrics_epoch = {key: [] for key in metricsdict.keys()}

        stage = 'train'
        if data_loader == dataloader_test:
            stage = 'test'

        for x, y in tqdm(data_loader):
            # print(model.encoder[0].weight)
            # print(torch.Tensor(x))
            y_prim = model.forward(x)

            loss = torch.mean(-y * torch.log(y_prim + 1e-8))

            if data_loader == dataloader_train:
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
                metricsdict[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 3)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    if epoch % 100 == 0:
        plts = []
        c = 0
        for key, value in metricsdict.items():
            plts += plt.plot(value, f'C{c}', label=key)
            ax = plt.twinx()
            c += 1

        plt.legend(plts, [it.get_label() for it in plts])
        plt.show()

        predicted = model.forward(dataset_test[:][0])
        predicted = np.argmax(predicted.cpu().data.numpy(), axis=1)
        expected = np.argmax(dataset_test[:][1].cpu().data.numpy(), axis=1)

        # confusion matrix and metrics
        c_d = metrics.confusion_matrix(expected, predicted)
        c_d = c_d/1.0
        accuracy = (c_d[0, 0] + c_d[1, 1])/np.sum(c_d)
        precision = c_d[0, 0]/(c_d[0, 0] + c_d[0, 1])
        recall = c_d[0, 0]/(c_d[0, 0] + c_d[1, 0])
        f1 = 2*precision*recall/(precision + recall)
        print("male")
        print("accuracy:", round(accuracy, 3), "precision:", round(precision, 3), "recall:", round(recall, 3), "f1:", round(f1, 3))
        accuracy = (c_d[0, 0] + c_d[1, 1])/np.sum(c_d)
        precision = c_d[1, 1]/(c_d[1, 1] + c_d[1, 0])
        recall = c_d[1, 1]/(c_d[1, 1] + c_d[0, 1])
        f1 = 2*precision*recall/(precision + recall)
        print()
        print("female")
        print("accuracy:", round(accuracy, 3), "precision:", round(precision, 3), "recall:", round(recall, 3), "f1:", round(f1, 3))
        print(c_d)
        c_d = c_d.astype(int)
        result = metrics.ConfusionMatrixDisplay(confusion_matrix = c_d, display_labels = ["Male", "Female"])
        result.plot()
        plt.show()