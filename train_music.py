'''
Authors : Equipe EEG
File: train_music.py
- Preparation of the Music Dataset
- Train the dataset with RNN Model

THE CODE IS INSPIRED BY THE FOLLOWING TUTORIALS:
1. Dataloader and Dataset in Pytorch: https://github.com/python-engineer/pytorchTutorial/blob/master/09_dataloader.py
2. Implementation of RNN Model: taken from: https://github.com/python-engineer/pytorch-examples/blob/master/rnn-lstm-gru/main.py
3. Fundamentals of the RNN Model: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html 
'''

# TRAINING MUSIC - GROUP 1
# python IFT3710/train_music.py /home/liuronni/projects/def-sponsor00/datasets/EEG/Music_eeg_raw 1

# TRAINING MUSIC - GROUP 2
# python IFT3710/train_music.py /home/liuronni/projects/def-sponsor00/datasets/EEG/Music_eeg_raw 2

from glob import glob
import os
import numpy as np
import sys
import random

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import math
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from eeg_music import music_preprocessing

print('EEG ANALYSIS: MUSIC THINKING \n')

# args[1] : path for dataset
print('arg[1] = ', sys.argv[1])
path = sys.argv[1]

# args[2]: group number (1 or 2)
group_number = sys.argv[2]

print('path =',path)

# Preprocessing
# Group selection
if int(group_number) == 1:
    # GROUP 1: Individiauls 0 to 34
    X, Y = music_preprocessing(path, 0, 35)
else:
    # GROUP 2: Individuals 36 to 70
    X, Y = music_preprocessing(path, 36, 71)

# We have 35 individuals for each type of labels. Therefore,
# for one group, we have 70 individuals x 20 epochs = 1400 epochs in total

# DIMENSION OF THE TENSOR
# The new preprocessed dataset's dimension: 14000 epochs x 64 electrodes x 750 timepoints

# Split the dataset into training - validation - test
# size1: ratio between training and validation/test datasets
# size2: ratio between validation and test datasets
def split_data(X,Y, size1, size2):
    print(' --------Split dataset -------- ')
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, Y, test_size = size1, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size = size2, random_state=42)

    print(np.shape(X_train),np.shape(y_train),np.shape(X_test),np.shape(y_test))

    print(X_train[:].shape)
    print(X_test[:].shape)
    print(X_val[:].shape)
    print('X_train.shape[0] = ',X_train.shape[0])
    print('X_train.shape[0][0] = ',np.shape(X_train[0][0]))
    print('X_test.shape[0] = ',X_test.shape[0])
    print('X_test.shape[0][0] = ',np.shape(X_test[0][0]))
    print('X_val.shape[0] = ',X_val.shape[0])
    print('X_val.shape[0][0] = ',np.shape(X_val[0][0]))
    print('\n')
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    
   
 # Dataloader in Pytorch --> to have appropriate format for model
def dataLoaderPytorch(x_train,x_val,x_test,input_size,seq_len):
    print(' -------- Dataloader in Torch -------- ')
    x_train_tensor = np.zeros((x_train.shape[0], 1, input_size,seq_len))
    x_test_tensor = np.zeros((x_test.shape[0], 1, input_size,seq_len))
    x_val_tensor = np.zeros((x_val.shape[0], 1, input_size,seq_len))

    for i in range(x_train_tensor.shape[0]):
        x_train_tensor[i,0,:,:] = x_train[i]
        
    for i in range(x_test_tensor.shape[0]):
        x_test_tensor[i,0,:,:] = x_test[i]

    for i in range(x_val_tensor.shape[0]):
        x_val_tensor[i,0,:,:] = x_val[i]
    
    return x_train_tensor, x_val_tensor, x_test_tensor


# Dataloader in pytorch
# Distribution of the epochs: 980 - 210 - 210 (training - validation - test)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, Y, 0.30, 0.5)
X_train_tensor, X_val_tensor, X_test_tensor = dataLoaderPytorch(X_train,X_val,X_test,64,750)

# Save the tensors into npy files for transfer learning
np.save('IFT3710/Datasets/music_train_' + group_number, X_train_tensor)
np.save('IFT3710/Datasets/music_val_' + group_number, X_val_tensor)
np.save('IFT3710/Datasets/music_test_' + group_number, X_test_tensor)

np.save('IFT3710/Datasets/music_train_labels_' + group_number, y_train)
np.save('IFT3710/Datasets/music_val_labels_' + group_number, y_val)
np.save('IFT3710/Datasets/music_test_labels_' + group_number, y_test)


# Classes for each dataset (training, validation, test)

# NOTE: The code for Dataset and Dataloaders were inspired by the tutorial:
# Link: https://github.com/python-engineer/pytorchTutorial/blob/master/09_dataloader.py 

class EEGTrain(Dataset):
    
    def __init__(self):
        self.x = torch.from_numpy(X_train_tensor).float()
        self.y = torch.from_numpy(y_train).long()
        self.n_samples = len(y_train)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
class EEGVal(Dataset):
    
    def __init__(self):
        self.x = torch.from_numpy(X_val_tensor).float()
        self.y = torch.from_numpy(y_val).long()
        self.n_samples = len(y_val)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
class EEGTest(Dataset):
    
    def __init__(self):
        self.x = torch.from_numpy(X_test_tensor).float()
        self.y = torch.from_numpy(y_test).long()
        self.n_samples = len(y_test)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
# RNN NETWORK
# Link: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

# The code for the RNN Model, and the training of the RNN Model is taken
# from the following tutorial about RNN, LSTM and GRU:
# Link: https://github.com/python-engineer/pytorch-examples/blob/master/rnn-lstm-gru/main.py 

class RNN(nn.Module):
    # Taken from: https://github.com/python-engineer/pytorch-examples/blob/master/rnn-lstm-gru/main.py
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out
    
    
# Function that calculates the accuracy of our predictions
def check_accuracy(loader, model, message):
    print(message)
    num_correct = 0
    num_samples = 0
    model.eval()
    
    # Source: https://github.com/python-engineer/pytorch-examples/blob/master/rnn-lstm-gru/main.py
    with torch.no_grad():
        for x, labels in loader:
            x = x.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device = device)
        
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy \
                    {float(num_correct)/float(num_samples)*100:2f}')
        
    model.train()


# Training the model

# HYPERPARAMETERS
# input_size = 64       #features : 64 electrodes
# sequence_length = 750 #sequence : 750 timepoints

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 64
sequence_length = 750 

# Best combination of hyperparameters: hidden_size = 64 and num_epochs = 20
num_classes = 2 
hidden_size = 64 
num_epochs = 20
batch_size = 8
learning_rate = 0.001
num_layers = 3


# Function that trains the RNN Model
def RNN_music():
    #------------ Phase train model -----------
    print(' --- Build model ---')

    # DATASETS
    train_data = EEGTrain()
    train_dl = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
    print('----- done DataLoader train_data')

    val_data = EEGVal()
    val_dl = DataLoader(dataset = val_data, batch_size = batch_size, shuffle = True)
    
    test_data = EEGTest()
    test_dl = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = True)
    print('----- done DataLoader train_data')
    print('----- done DataLoader val_data')
    print('\n')    
    print('-----------------------')
    
    #----- Create model ----
    print('--* Create model *--')    
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
    print('--* done *--')
    print('-----------------------')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    # Train the model
    # Taken from: https://github.com/python-engineer/pytorch-examples/blob/master/rnn-lstm-gru/main.py
    print('--* Train model *--')    
    n_total_steps = len(train_dl)
    for epoch in range(num_epochs):
        for i, (x, labels) in tqdm(enumerate(train_dl)):  
            x = x.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device=device)

            outputs = model(x)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print('--* done *--')
    print('-----------------------')
    return train_dl, val_dl, test_dl, model

train_dl, val_dl, test_dl, model = RNN_music()

print('-- Hyperparameters : ')
print('hidden_size = ',hidden_size)
print('num_epochs = ',num_epochs)
print('batch_size = ',batch_size)

# ACCURACY FOR THE GIVEN GROUP (1 OR 2): Training/validation/test datasets
print(f'Accuracy for Group {group_number}')
check_accuracy(train_dl, model, 'Checking accuracy on training data')
check_accuracy(val_dl, model,'Checking accuracy on val data')
check_accuracy(test_dl, model,'Checking accuracy on test data')
              
print('--* done *--')
print('-----------------------')