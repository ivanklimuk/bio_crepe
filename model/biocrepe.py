import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, \
                            precision_score, \
                            recall_score, \
                            fbeta_score, \
                            roc_auc_score, \
                            confusion_matrix as cm


class BioCrepe(nn.Module):
    def __init__(self,
                 vocabulary_size,
                 channels,
                 kernel_sizes,
                 pooling_sizes,
                 linear_size,
                 output_size,
                 dropout
                ):
        super(BioCrepe, self).__init__()

        # define the model parameters
        self.vocabulary_size = vocabulary_size
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.pooling_sizes = pooling_sizes
        self.linear_size = linear_size
        self.output_size = output_size
        self.dropout = dropout

        # define the model layers
        self.convolution_0 = nn.Conv1d(in_channels=self.vocabulary_size,
                                       out_channels=self.channels[0],
                                       kernel_size=self.kernel_sizes[0])
        nn.init.normal_(self.convolution_0.weight, mean=0, std=0.1)
        self.max_pooling_0 = nn.MaxPool1d(kernel_size=self.pooling_sizes[0])
        self.activation_0 = nn.ReLU(inplace=False)

        self.convolution_1 = nn.Conv1d(in_channels=self.channels[0],
                                       out_channels=self.channels[1],
                                       kernel_size=self.kernel_sizes[1])
        nn.init.normal_(self.convolution_1.weight, mean=0, std=0.1)
        self.max_pooling_1 = nn.MaxPool1d(kernel_size=self.pooling_sizes[1])
        self.activation_1 = nn.ReLU(inplace=False)

        self.convolution_2 = nn.Conv1d(in_channels=self.channels[1],
                                       out_channels=self.channels[2],
                                       kernel_size=self.kernel_sizes[2])
        nn.init.normal_(self.convolution_2.weight, mean=0, std=0.1)
        self.max_pooling_2 = nn.MaxPool1d(kernel_size=self.pooling_sizes[2])
        self.activation_2 = nn.ReLU(inplace=False)

        self.convolution_3 = nn.Conv1d(in_channels=self.channels[2],
                                       out_channels=self.channels[3],
                                       kernel_size=self.kernel_sizes[3])
        nn.init.normal_(self.convolution_3.weight, mean=0, std=0.1)
        self.max_pooling_3 = nn.MaxPool1d(kernel_size=self.pooling_sizes[3])
        self.activation_3 = nn.ReLU(inplace=False)

        self.linear_4 = nn.Linear(self.linear_size, self.linear_size)
        self.batch_norm_4 = nn.BatchNorm1d(self.linear_size)
        self.activation_4 = nn.ReLU(inplace=False)
        self.drop_out_4 = nn.Dropout(p=self.dropout, inplace=False)

        self.linear_5 = nn.Linear(self.linear_size, self.linear_size)
        self.batch_norm_5 = nn.BatchNorm1d(self.linear_size)
        self.activation_5 = nn.ReLU(inplace=False)
        self.drop_out_5 = nn.Dropout(p=self.dropout, inplace=False)

        self.linear_output = nn.Linear(self.linear_size, self.output_size)
        self.activation_output = nn.Sigmoid()

    def one_hot(self, x):
        x = x.numpy().astype(int)
        one_hot_encoded = np.zeros((x.shape[0], self.vocabulary_size, x.shape[1]))

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                one_hot_encoded[i, x[i, j] - 1, j] = 1 if x[i, j] > 0 else 0

        return torch.FloatTensor(one_hot_encoded)

    def forward(self, s):
        # one hot encoding
        s = self.one_hot(s)
        # convolution x 4
        s = self.activation_0(self.max_pooling_0(self.convolution_0(s)))
        s = self.activation_1(self.max_pooling_1(self.convolution_1(s)))
        s = self.activation_2(self.max_pooling_2(self.convolution_2(s)))
        s = self.activation_3(self.max_pooling_3(self.convolution_3(s)))
        # flatten before the FC layer
        s = s.view(s.size(0), -1)
        # linear + batch_norm + dropout x 2
        s = self.drop_out_4(self.activation_4(self.batch_norm_4(self.linear_4(s))))
        s = self.drop_out_5(self.activation_5(self.batch_norm_5(self.linear_5(s))))
        # sigmoid output
        s = self.activation_output(self.linear_output(s))

        return s


def accuracy(y_true, y_predicted):
    return accuracy_score(y_true, np.round(y_predicted))


def precision(y_true, y_predicted):
    return precision_score(y_true, np.round(y_predicted), average='macro')


def recall(y_true, y_predicted):
    return recall_score(y_true, np.round(y_predicted), average='macro')


def f1(y_true, y_predicted):
    return fbeta_score(y_true, np.round(y_predicted), average='macro', beta=1)


def confusion_matrix(y_true, y_predicted):
    return cm(y_true, np.round(y_predicted))


def roc(y_true, y_predicted):
    return roc_auc_score(y_true, y_predicted)


metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1
}
