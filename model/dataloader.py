import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


class DataLoader(object):
    def __init__(self,
                 data_path,
                 max_length,
                 labels_path=None,
                 categorical=False,
                 header=None,
                 lower=False,
                 test_size=0.15
                 ):
        self.data_path = data_path
        self.labels_path = labels_path
        self.categorical = categorical
        self.max_length = max_length
        self.header = header
        self.lower = lower

    def _load_data(self):
        data = np.array(pd.read_csv(self.data_path, header=self.header))
        if self.labels_path:
            labels = np.array(pd.read_csv(self.labels_path, header=self.header))
        else:
            labels, data = data[:, 0], data[:, 1]

        if self.categorical:
            labels = np.vstack((
                (labels == 1).astype(int),
                (labels == 0).astype(int)
            )).T
        else:
            labels = labels.reshape(-1, 1)

        self.text_data = data
        self.labels = labels

    def create_vocabulary(self):
        self.alphabet = ['a', 'c', 'g', 't']
        vocabulary = {character: number + 1  # + 1 to avoid further confusion with 0s
            for number, character in enumerate(self.alphabet)}
        self.vocabulary = vocabulary
        self.vocabulary_size = len(vocabulary)

    def text_to_array(self, texts):
        array = np.zeros((len(texts), self.max_length), dtype=np.int)
        for row, line in enumerate(texts):
            if self.lower:
                line = line.lower()
            for column in range(min([len(line), self.max_length])):
                array[row, column] = self.vocabulary.get(line[column], -1)  # replace with -1
        return array

    def load_data(self, split=True):
        self._load_data()
        self.create_vocabulary()
        self.data = self.text_to_array(self.text_data)

        self.data = self.data.astype(np.float)
        self.labels = self.labels.astype(np.float)

        if split:
            return train_test_split(self.data, self.labels, random_state=42, test_size=0.15)
        else:
            return self.data, self.labels


def data_iterator(data, labels, batch_size, shuffle=True):
    while True:
        if shuffle:
            shuf = np.random.permutation(len(data))
            data = data[shuf]
            labels = labels[shuf]
        for i in range(0, len(data), batch_size):
            yield torch.autograd.Variable(torch.from_numpy(data[i:i + batch_size]).float()), \
                  torch.autograd.Variable(torch.from_numpy(labels[i:i + batch_size]).float())
