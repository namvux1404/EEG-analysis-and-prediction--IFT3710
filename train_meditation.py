'''
Authors : Equipe EEG
Fichier pour training le model RNN pour dataset Music_meditation
Last updated : 15-04-2022
'''
#file python to train the dataset of eeg meditation
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
from eeg_meditation import preprocessing

print('EEG ANALYSIS: MEDITATION VS THINKING \n')
#args[1] : path for dataset

print('arg[0] = ', sys.argv[1])

path = sys.argv[1]

print('path =', path)

X, Y = preprocessing(path) # Etape pour preprocessing dataset music

#--- split dataset
print(' --------Split dataset -------- ')
X_train, X_valtest, y_train, y_valtest = train_test_split(X, Y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)

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

 #------------- Dataloader in Torch --> to have appropriate format for model -------
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

#--- Dataloader in pytorch
X_train_tensor, X_val_tensor, X_test_tensor = dataLoaderPytorch(X_train,X_val,X_test,64,750)

class EEGTrain(Dataset):
    
    def __init__(self):
        #data loading
        self.x = torch.from_numpy(X_train_tensor).float()
        self.y = torch.from_numpy(y_train).long()
        self.n_samples = len(y_train)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        # len(dataset)
        return self.n_samples
    
class EEGVal(Dataset):
    
    def __init__(self):
        #data loading
        self.x = torch.from_numpy(X_val_tensor).float()
        self.y = torch.from_numpy(y_val).long()
        self.n_samples = len(y_val)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        # len(dataset)
        return self.n_samples
    
class EEGTest(Dataset):
    
    def __init__(self):
        #data loading
        self.x = torch.from_numpy(X_test_tensor).float()
        self.y = torch.from_numpy(y_test).long()
        self.n_samples = len(y_test)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        # len(dataset)
        return self.n_samples
    
# NETWORK
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        # x -> (batch_size, sequence_length, input_size)

        # or:
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.rnn(x, h0)
        # out -> (batch_size, sequence_length, hidden_size)

        #out = out[:, -1, :]
        out = out.reshape(out.shape[0], -1)
        # out -> (N, 129)
        out = self.fc(out)
        return out
    
def check_accuracy(loader, model, message):
    print(message)
    
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, labels in loader:
            #x = x.to(device = device).squeeze(1)
            x = x.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device = device)
        
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy \
                    {float(num_correct)/float(num_samples)*100:2f}')
        
    model.train()

#create device for training model
# HYPERPARAMETERS
#input_size = 129 #features : 129 electrodes
#sequence_length = 750 #sequence : 750 timepoints
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 64 #features : 129 electrodes
sequence_length = 750 #sequence : 500 timepoints

num_classes = 2 #classification 
hidden_size = 64 #donne meilleur 
num_epochs = 20
batch_size = 8 #number of examples in 1 forward pass --> 4 epochs
learning_rate = 0.001
num_layers = 3
print('----- done hyperparameters')

def RNN_meditation():

 
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
    print('--* Train model *--')    
    n_total_steps = len(train_dl)
    for epoch in range(num_epochs):
        for i, (x, labels) in tqdm(enumerate(train_dl)):  
            # origin shape: [N, 1, 500,129]
            # resized: [N, 500, 129]
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

train_dl, val_dl, test_dl, model = RNN_meditation()

check_accuracy(train_dl, model, 'Checking accuracy on training data')
check_accuracy(val_dl, model,'Checking accuracy on val data')
check_accuracy(test_dl, model,'Checking accuracy on test data')
              
print('--* done *--')
print('-----------------------')