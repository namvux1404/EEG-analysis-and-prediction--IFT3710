# python IFT3710/transfer_learning.py
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader

print('TRANSFER LEARNING:')
# Preprocessing for the two datasets
# Dataset A: Music
# Dataset B: Meditation

################################
# PART I: TENSORS FOR MUSIC
################################

XA_train = np.load(f'IFT3710/Datasets/music_train_1.npy')
XA_val = np.load(f'IFT3710/Datasets/music_val_1.npy')
XA_test = np.load(f'IFT3710/Datasets/music_test_1.npy')

yA_train = np.load(f'IFT3710/Datasets/music_train_labels_1.npy')
yA_val = np.load(f'IFT3710/Datasets/music_val_labels_1.npy')
yA_test = np.load(f'IFT3710/Datasets/music_test_labels_1.npy')

print('SHAPES OF TENSORS FOR MUSIC')
print(np.shape(XA_train), np.shape(XA_val), np.shape(XA_test))
print(np.shape(yA_train), np.shape(yA_val), np.shape(yA_test))


################################
# PART II: DATALOADERS FOR MUSIC AND RNN MODEL
################################
class EEGATrain(Dataset):
    def __init__(self):
        #data loading
        self.x = torch.from_numpy(XA_train).float()
        self.y = torch.from_numpy(yA_train).long()
        self.n_samples = len(yA_train)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        # len(dataset)
        return self.n_samples
    
class EEGAVal(Dataset):
    def __init__(self):
        #data loading
        self.x = torch.from_numpy(XA_val).float()
        self.y = torch.from_numpy(yA_val).long()
        self.n_samples = len(yA_val)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        # len(dataset)
        return self.n_samples
    
class EEGATest(Dataset):
    def __init__(self):
        #data loading
        self.x = torch.from_numpy(XA_test).float()
        self.y = torch.from_numpy(yA_test).long()
        self.n_samples = len(yA_test)
        
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 64 #features : 129 electrodes
sequence_length = 750 #sequence : 500 timepoints

num_classes = 2 #classification 
hidden_size = 64 #donne meilleur 
num_epochs = 20
batch_size = 8 #number of examples in 1 forward pass --> 4 epochs
learning_rate = 0.001
num_layers = 3
print('----- Done hyperparameters')


# Build the model
print('--- BUILDING THE MODEL FOR MUSIC... ---')

music_train_data = EEGATrain()
music_train_dl = DataLoader(dataset = music_train_data, batch_size = batch_size, shuffle = True)
print('----- done DataLoader train_data')

music_val_data = EEGAVal()
music_val_dl = DataLoader(dataset = music_val_data, batch_size = batch_size, shuffle = True)
print('----- done DataLoader val_data')

music_test_data = EEGATest()
music_test_dl = DataLoader(dataset = music_test_data, batch_size = batch_size, shuffle = True)
print('----- done DataLoader test_data')
print('\n')

print('--* CREATE MODEL FOR MUSIC *--')
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
print('--* done *--')
print('-----------------------')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

# Train the model
print('--* Train model *--')    
n_total_steps = len(music_train_dl)
for epoch in range(num_epochs):
    for i, (x, labels) in tqdm(enumerate(music_train_dl)):  
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
print('-- Les parametres : ')
print('hidden_size = ',hidden_size)
print('num_epochs = ',num_epochs)
print('batch_size = ',batch_size)

print('ACCURACY FOR MUSIC RNN')
check_accuracy(music_train_dl, model, 'Checking accuracy on training data')
check_accuracy(music_val_dl, model,'Checking accuracy on val data')
check_accuracy(music_test_dl, model,'Checking accuracy on test data')

print('--* baseline rnn done *--')
print('-----------------------')


################################
# PART III: TENSORS FOR MEDITATION
################################

'''
XB_train = np.load('IFT3710/Datasets/med_train.npy')
XB_val = np.load('IFT3710/Datasets/med_val.npy')
XB_test = np.load('IFT3710/Datasets/med_test.npy')

yB_train = np.load('IFT3710/Datasets/med_train_labels.npy')
yB_val = np.load('IFT3710/Datasets/med_val_labels.npy')
yB_test = np.load('IFT3710/Datasets/med_test_labels.npy')
'''

XB_train = np.load(f'IFT3710/Datasets/music_train_2.npy')
XB_val = np.load(f'IFT3710/Datasets/music_val_2.npy')
XB_test = np.load(f'IFT3710/Datasets/music_test_2.npy')

yB_train = np.load(f'IFT3710/Datasets/music_train_labels_2.npy')
yB_val = np.load(f'IFT3710/Datasets/music_val_labels_2.npy')
yB_test = np.load(f'IFT3710/Datasets/music_test_labels_2.npy')


print('SHAPES OF TENSORS FOR MEDITATION')
print(np.shape(XB_train), np.shape(XB_val), np.shape(XB_test))
print(np.shape(yB_train), np.shape(yB_val), np.shape(yB_test))


################################
# PART IV: DATALOADERS FOR MEDITATION
################################

class EEGBTrain(Dataset):
    def __init__(self):
        #data loading
        self.x = torch.from_numpy(XB_train).float()
        self.y = torch.from_numpy(yB_train).long()
        self.n_samples = len(yB_train)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        # len(dataset)
        return self.n_samples
    
class EEGBVal(Dataset):
    def __init__(self):
        #data loading
        self.x = torch.from_numpy(XB_val).float()
        self.y = torch.from_numpy(yB_val).long()
        self.n_samples = len(yB_val)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        # len(dataset)
        return self.n_samples
    
class EEGBTest(Dataset):
    def __init__(self):
        #data loading
        self.x = torch.from_numpy(XB_test).float()
        self.y = torch.from_numpy(yB_test).long()
        self.n_samples = len(yB_test)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        # len(dataset)
        return self.n_samples

################################
# PART V: MODEL WITH TRANSFER LEARNING
################################

# Build the model
print('--- BUILDING THE MODEL FOR MEDITATION WITH TRANSFER LEARNING... ---')

med_train_data = EEGBTrain()
med_train_dl = DataLoader(dataset = med_train_data, batch_size = batch_size, shuffle = True)
print('----- done DataLoader train_data')

med_val_data = EEGBVal()
med_val_dl = DataLoader(dataset = med_val_data, batch_size = batch_size, shuffle = True)
print('----- done DataLoader val_data')

med_test_data = EEGBTest()
med_test_dl = DataLoader(dataset = med_test_data, batch_size = batch_size, shuffle = True)
print('----- done DataLoader test_data')
print('\n')

# Freeze the parameters
# Transfer Learning
freeze_choice = input("Would you like to freeze all the parameters (1) or only the first layer (2)?")

if int(freeze_choice) == 1:
    print('CHOICE (1): FREEZE ALL THE PARAMETERS')
    
    for param in model.parameters():
        param.requires_grad = False
else:
    print('CHOICE (2): FREEZE ONLY THE FIRST LAYER')
    
    counter = 0
    for child in model.children():
        counter += 1
        if counter == 1:
            for param in child.parameters():
                param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

# Train the model with freezed parameters
print('--* Train model *--')    
n_total_steps = len(med_train_dl)
for epoch in range(num_epochs):
    for i, (x, labels) in tqdm(enumerate(med_train_dl)):  
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

print('ACCURACY FOR MEDITATION RNN WITH TRANSFER LEARNING')
check_accuracy(med_train_dl, model, 'Checking accuracy on training data')
check_accuracy(med_val_dl, model,'Checking accuracy on val data')
check_accuracy(med_test_dl, model,'Checking accuracy on test data')

print('--* done *--')
print('-----------------------')