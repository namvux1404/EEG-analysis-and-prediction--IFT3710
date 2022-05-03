# python IFT3710/train_meditation.py /home/liuronni/projects/def-sponsor00/datasets/EEG/Med_eeg_raw
'''
Authors : EEG Team
File: transfer_learning.py
- Apply transfer learning from one dataset to another

THE CODE IS INSPIRED BY THE FOLLOWING TUTORIALS:
1. Dataloader and Dataset in Pytorch: https://github.com/python-engineer/pytorchTutorial/blob/master/09_dataloader.py
2. Implementation of RNN Model: taken from: https://github.com/python-engineer/pytorch-examples/blob/master/rnn-lstm-gru/main.py
3. Fundamentals of the RNN Model: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html 
'''

from glob import glob
import os
import numpy as np
import sys
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from eeg_meditation import med_preprocessing
from logisticsRegression import logRegression


# args[1] : path for dataset

print('arg[1] = ', sys.argv[1])

path = sys.argv[1]

print('path =', path)

# Etape pour preprocessing dataset music
X, Y, features, group = med_preprocessing(path)

print('EEG ANALYSIS: MEDITATION VS THINKING \n')

###########################################
######## Model Logistics Regression #######
###########################################
print('---- Model Logistics Regression -----')

param_grid = {'clf__C': [0.1, 0.5, 0.7, 1, 3, 5, 7, 10, 15, 20, 50]}

accuracy, best_param, best_score = logRegression(
    Y, features, group, param_grid)

print('Param = ', param_grid)
print('Accuracy = ', accuracy)
print('Best_param - Best_score = ', best_param,
      '-', best_score)
print('---------------------------------------')


########################################
############### Model RNN ##############
########################################
print('---- Model RNN -----')


# -------------------------- Functions predefine for RNN Model -----------------

# Split the dataset into training - validation - test
# size1: ratio between training and validation/test datasets
# size2: ratio between validation and test datasets


def split_data(X, Y, size1, size2):
    print(' --------Split dataset -------- ')
    X_train, X_valtest, y_train, y_valtest = train_test_split(
        X, Y, test_size=size1, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_valtest, y_valtest, test_size=size2, random_state=42)

    print(np.shape(X_train), np.shape(y_train),
          np.shape(X_test), np.shape(y_test))

    print(X_train[:].shape)
    print(X_test[:].shape)
    print(X_val[:].shape)
    print('X_train.shape[0] = ', X_train.shape[0])
    print('X_train.shape[0][0] = ', np.shape(X_train[0][0]))
    print('X_test.shape[0] = ', X_test.shape[0])
    print('X_test.shape[0][0] = ', np.shape(X_test[0][0]))
    print('X_val.shape[0] = ', X_val.shape[0])
    print('X_val.shape[0][0] = ', np.shape(X_val[0][0]))

    return X_train, X_val, X_test, y_train, y_val, y_test

# ------------- Dataloader in Torch --> to have appropriate format for model -------


# Function to load data in appropriate shape for Pytorch
def dataLoaderPytorch(x_train, x_val, x_test, input_size, seq_len):
    print(' -------- Dataloader in Torch -------- ')
    x_train_tensor = np.zeros((x_train.shape[0], 1, input_size, seq_len))
    x_test_tensor = np.zeros((x_test.shape[0], 1, input_size, seq_len))
    x_val_tensor = np.zeros((x_val.shape[0], 1, input_size, seq_len))

    for i in range(x_train_tensor.shape[0]):
        x_train_tensor[i, 0, :, :] = x_train[i]

    for i in range(x_test_tensor.shape[0]):
        x_test_tensor[i, 0, :, :] = x_test[i]

    for i in range(x_val_tensor.shape[0]):
        x_val_tensor[i, 0, :, :] = x_val[i]

    return x_train_tensor, x_val_tensor, x_test_tensor


class EEGTrain(Dataset):  # class for EEG train

    def __init__(self):
        # data loading
        self.x = torch.from_numpy(X_train_tensor).float()
        self.y = torch.from_numpy(y_train).long()
        self.n_samples = len(y_train)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples


class EEGVal(Dataset):  # class for EEG Validation

    def __init__(self):
        # data loading
        self.x = torch.from_numpy(X_val_tensor).float()
        self.y = torch.from_numpy(y_val).long()
        self.n_samples = len(y_val)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples


class EEGTest(Dataset):  # class for EEG Test

    def __init__(self):
        # data loading
        self.x = torch.from_numpy(X_test_tensor).float()
        self.y = torch.from_numpy(y_test).long()
        self.n_samples = len(y_test)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples


class RNN(nn.Module):  # Class for RNN
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size,
                          num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)

        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)

        return out


# Function to check accuracy of the model
def check_accuracy(loader, model, message):
    print(message)

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, labels in loader:

            x = x.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy \
                    {float(num_correct)/float(num_samples)*100:2f}')

    model.train()


def RNN_meditation():  # Principal function to create model RNN and train

    # ------------ Phase train model -----------
    print(' --- Build model ---')

    # DATASETS
    train_data = EEGTrain()
    train_dl = DataLoader(dataset=train_data,
                          batch_size=batch_size, shuffle=True)
    print('----- done DataLoader train_data')

    val_data = EEGVal()
    val_dl = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

    test_data = EEGTest()
    test_dl = DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=True)
    print('----- done DataLoader train_data')
    print('----- done DataLoader val_data')
    print('\n')
    print('-----------------------')
    # ----- Create model ----
    print('--* Create model *--')
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
    print('--* done *--')
    print('-----------------------')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print('--* Train model *--')
    n_total_steps = len(train_dl)
    for epoch in range(num_epochs):
        for i, (x, labels) in tqdm(enumerate(train_dl)):

            x = x.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device=device)

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print('--* done *--')
    print('-----------------------')
    return train_dl, val_dl, test_dl, model

# --------------------------- End of functions predefine ---------------------


# --- Split dataset
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, Y, 0.30, 0.5)

#--- Dataloader in pytorch
X_train_tensor, X_val_tensor, X_test_tensor = dataLoaderPytorch(
    X_train, X_val, X_test, 64, 750)


# Save the tensors into npy files for transfer learning
np.save('IFT3710/Datasets/med_train', X_train_tensor)
np.save('IFT3710/Datasets/med_val', X_val_tensor)
np.save('IFT3710/Datasets/med_test', X_test_tensor)

np.save('IFT3710/Datasets/med_train_labels', y_train)
np.save('IFT3710/Datasets/med_val_labels', y_val)
np.save('IFT3710/Datasets/med_test_labels', y_test)

# create device for training model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# HYPERPARAMETERS
input_size = 64  # features : 64 electrodes
sequence_length = 750  # sequence : 750 timepoints

num_classes = 2  # classification
hidden_size = 64  # donne meilleur
num_epochs = 20
batch_size = 8  # number of examples in 1 forward pass
learning_rate = 0.001
num_layers = 3
print('----- done hyperparameters')

train_dl, val_dl, test_dl, model = RNN_meditation()

check_accuracy(train_dl, model, 'Checking accuracy on training data')
check_accuracy(val_dl, model, 'Checking accuracy on val data')
check_accuracy(test_dl, model, 'Checking accuracy on test data')

print('--* done *--')
print('-----------------------')
