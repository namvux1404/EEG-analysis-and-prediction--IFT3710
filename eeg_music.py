from glob import glob
import os
import mne
import numpy as np
import sys
import random

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import math

print('EEG ANALYSIS: MUSIC THINKING \n')

#args[1] : path for dataset

print('arg[0] = ', sys.argv[1])

path = sys.argv[1]

print('path =',path)

#all_eeg_path = glob('Music Thinking/sub-*/ses-*/eeg/*.set')
#need to sort file to order
all_eeg_path = sorted(glob(path + '/*.set')) 

print(f'Number of individuals: {len(all_eeg_path)}')

#all_eeg_path = sorted(glob.glob('*.png'))

#for i in range(30):
#    print('all_eeg_path =',all_eeg_path[i])
#print('all_eeg_path[1] = ',all_eeg_path[1])
#print('all_eeg_path[2] = ',all_eeg_path[2])
#import pdb; pdb.set_trace()

# Array for the labels
#behaviour_data = np.genfromtxt('Music Thinking/stimuli/Behavioural_data.txt')
behaviour_data = np.genfromtxt(path + '/stimuli_Behavioural_data.txt')
behaviour_data = behaviour_data[1:]

labels = behaviour_data[:, 2]
for i in range(labels.shape[0]):
    # Enjoyment level 1 or 2 -> enjoy the most
    if labels[i] <= 2:
        labels[i] = 1

    # Neutral or does not enjoy (enjoyment levels 3 to 5)
    else:
        labels[i] = 0

# The proportion of people liking/disliking the music is relatively the same.
print(f'Number of people liking the music: {np.count_nonzero(labels == 1)}')  # LIKE
print(f'Number of people disliking the music: {np.count_nonzero(labels == 0)}')  # DISLIKE

# Associate each path to the corresponding class (binary classification)
like_path = [] 
dislike_path = []

for i in range(labels.shape[0]):
    if labels[i] == 1:
        like_path = np.append(like_path, all_eeg_path[i])
    else:
        dislike_path = np.append(dislike_path, all_eeg_path[i])

print(f'\nLet\'s confirm that the length of like_path and dislike_path are equal to the number of labels')
print(f'Is like_path same length as qty of labels? {len(like_path) == np.count_nonzero(labels == 1)}')
print(f'Is dislike_path same length as qty of labels? {len(dislike_path) == np.count_nonzero(labels == 0)}')

print('----------------')

#test 
like_path = like_path[0:35]
dislike_path = dislike_path[0:36]
print('shape of like and dislike path : ',like_path.shape[0], dislike_path.shape[0]) 

print('----------------')
# Function that reads EEG Data (set extension file)
def read_set_data(path):
    print('---- File path -----')
    print('Begin read with path = ',path)
    music_data = mne.io.read_raw_eeglab(path, preload=False)
    
    # Preprocessing (Filtering)
    # music_data.filter(l_freq = 0.1, h_freq = 60)
    #import pdb; pdb.set_trace()
    
    epochs = mne.make_fixed_length_epochs(music_data, duration=3, overlap=1,preload = False)
    music_array = epochs.get_data()
    #shape music_array = 134x129x750
    
    music_array = music_array[:,:,:750]
    #shape music_array = 134x129x750
    
    #keep only 4 epochs for training ?
    number_epochs = 10
    array_epochs = np.empty(number_epochs, dtype = object)

    for i in range(number_epochs):
        chosen_number = random.randint(0, music_array.shape[0]-1)
        #print(chosen_number)
        
        #array_epochs[i] = np.transpose(music_array[chosen_number])   #500x129
        array_epochs[i] = music_array[chosen_number]                #129x500
        
    print(f'Dimensions of the tensor: {array_epochs[0].shape}')
    
    #shape array_epochs : 129x750
    #129 electrodes x 750 time points
    return array_epochs


#------- read all eeg files ------ #
print('------ Step read all eeg files --------')

## NOTE : dislike_path[20] error file, dislike_path[33]?

#music_array_1 = read_set_data(dislike_path[33])
#print('music_array_1 shape =',np.shape(music_array_1))

#music_array_2 = read_set_data(like_path[1])
#print('music_array_2 shape =',np.shape(music_array_2))

random.seed(0)

like_epoch_array = np.empty((len(like_path)), dtype = object)
dislike_epoch_array = np.empty((len(dislike_path)), dtype = object)
#like_epoch_array = []
#dislike_epoch_array = []

    
print('shape of like and dislike epoch array : ',np.shape(like_epoch_array), np.shape(dislike_epoch_array))

for i in range(len(like_path)):
    print('counting i = ',i)
    like_epoch_array[i] = read_set_data(like_path[i])
    #like_epoch_array.append(read_set_data(like_path[i]))
    
for i in range(len(dislike_path)):
    if (i == 20) : continue
    else:
        print('counting i = ',i)
        dislike_epoch_array[i] = read_set_data(dislike_path[i])
        #dislike_epoch_array.append(read_set_data(dislike_path[i]))
        

dislike_epoch_array = np.delete(dislike_epoch_array, 20, 0)

print('\n')  
print('shape of like_epoch_array and like_epoch_array[0]: ',np.shape(like_epoch_array), np.shape(like_epoch_array[0][1]))
print('shape of dislike_epoch_array and dislike_epoch_array[0]: ',np.shape(dislike_epoch_array), np.shape(dislike_epoch_array[0][1]))
print('\n')
#-----------
print(' --- Create label array ---- ')
# Assign the labels for each epoch
like_epoch_labels = np.empty((len(like_epoch_array)), dtype = object)
dislike_epoch_labels = np.empty((len(dislike_epoch_array)), dtype = object)

for i in range(len(like_epoch_array)):
    like_epoch_labels[i] = len(like_epoch_array[i]) * [1]
    
for i in range(len(dislike_epoch_array)):
    dislike_epoch_labels[i] = len(dislike_epoch_array[i]) * [0]
    
print('shape like_epoch_labels and dislike_epoch_labels = ',np.shape(like_epoch_labels), np.shape(dislike_epoch_labels))

print('\n')
#------------- Split dataset -------
print(' --------Split dataset -------- ')
X_train = np.hstack(np.append(like_epoch_array[0:20], dislike_epoch_array[0:20]))
X_val = np.hstack(np.append(like_epoch_array[20:], dislike_epoch_array[20:]))

y_train = np.hstack(np.append(like_epoch_labels[0:20], dislike_epoch_labels[0:20]))
y_val = np.hstack(np.append(like_epoch_labels[20:], dislike_epoch_labels[20:]))

print(X_train[:].shape)
print(X_val[:].shape)

print('X_train.shape[0] = ',X_train.shape[0])
print('X_train.shape[0][0] = ',np.shape(X_train[0][0]))
print('X_val.shape[0] = ',X_val.shape[0])
print('X_val.shape[0][0] = ',np.shape(X_train[0][0]))

print('\n')
#------------- Dataloader in Torch --> to have appropriate format for model -------
print(' -------- Dataloader in Torch -------- ')
X_train_tensor = np.zeros((X_train.shape[0], 1, 129,750))
X_val_tensor = np.zeros((X_val.shape[0], 1, 129,750))

for i in range(X_train_tensor.shape[0]):
    X_train_tensor[i,0,:,:] = X_train[i]
    
for i in range(X_val_tensor.shape[0]):
    X_val_tensor[i,0,:,:] = X_val[i]

print('Tensor shape X_train, X_val, Y_train, Y_val = ',X_train_tensor.shape, X_val_tensor.shape, y_train.shape, y_val.shape)

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
print('\n')    
#------------ Phase train model -----------
print(' --- Build model ---')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# HYPERPARAMETERS
#input_size = 129 #features : 129 electrodes
#sequence_length = 500 #sequence : 500 timepoints

input_size = 129 #features : 129 electrodes
sequence_length = 750 #sequence : 500 timepoints

num_classes = 2 #classification 
hidden_size = 256
num_epochs = 3
batch_size = 8 #number of examples in 1 forward pass --> 4 epochs
learning_rate = 0.001

num_layers = 2

print('----- done hyperparameters')

# DATASETS
train_data = EEGTrain()
train_dl = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
print('----- done DataLoader train_data')

val_data = EEGVal()
val_dl = DataLoader(dataset = val_data, batch_size = batch_size, shuffle = True)
print('----- done DataLoader val_data')
print('\n')    

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
        #out, _ = self.gru(x, h0)
        #out, _ = self.lstm(x, h0)
        # out -> (batch_size, sequence_length, hidden_size)
        # out -> (N, 129, 128) ->> NOTE: CHANGE SEQUENCE_LENGTH AS 750 LATER (TRANPOSE THE TENSOR)
        
        #out = out[:, -1, :]
        out = out.reshape(out.shape[0], -1)
        # out -> (N, 129)
        out = self.fc(out)
        return out
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
    for i, (images, labels) in enumerate(train_dl):  
        # origin shape: [N, 1, 500,129]
        # resized: [N, 500, 129]
        images = images.reshape(-1, sequence_length, input_size).to(device)
        #images = images.to(device=device).squeeze(1)
        labels = labels.to(device=device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('--* done *--')
print('-----------------------')
'''
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
print('--* Train model *--')  
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in val_dl:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        
        _, predicted = torch.max(outputs.data, 1)
        
        print(predicted, labels)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')
'''
def check_accuracy(loader, model,message):
    '''
    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')
    '''
    print(message)
    
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for images, labels in loader:
            #images = images.to(device = device).squeeze(1)
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device = device)
        
            scores = model(images)
            _, predictions = scores.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy \
                    {float(num_correct)/float(num_samples)*100:2f}')
        
    model.train()
              
check_accuracy(train_dl, model, 'Checking accuracy on training data')
check_accuracy(val_dl, model,'Checking accuracy on test data')
              
print('--* done *--')
print('-----------------------')
# Plot visualization
#ex = mne.io.read_raw_eeglab(all_eeg_path[0], preload = True)
#ex.plot()
#ex.plot_psd()

# Test with the first EEG
# We have a tensor of (45, 129, 1000)
# test = read_set_data(all_eeg_path[0])


''' 
# Separate continuous signals into epochs (segments) with equal time points
like_epochs_array = [read_set_data(i) for i in like_path]
dislike_epochs_array = [read_set_data(i) for i in dislike_path]

print(f'Verification of the dimensions of 1st individual: {like_epochs_array[0].shape}')

# Assign the labels for each epoch
like_epoch_labels = np.empty((len(like_path)), dtype=object)
dislike_epoch_labels = np.empty((len(dislike_path)), dtype=object)

for i in range(len(like_path)):
    like_epoch_labels[i] = len(like_epochs_array[i]) * [1]

for i in range(len(dislike_path)):
    dislike_epoch_labels[i] = len(dislike_epochs_array[i]) * [0]

print(len(like_epoch_labels), len(dislike_epoch_labels))

# Assign the individuals for each epoch
data_list = like_epochs_array + dislike_epochs_array
label_list = like_epoch_labels + dislike_epoch_labels

person_list = np.empty(len(all_eeg_path)), dtype = object)
for i in range(len(person_list)):
    person_list[i] = len(data_list[i]) * [i]
    
print(len(person_list))

# TODO: Split the dataset into three parts: Training - Validation - Test
'''
