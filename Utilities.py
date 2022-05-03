import torch
import h5py
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
import dask.array as da
from time import time
import random
import torch.nn.functional as F
# import EarlyStopping
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq,rfft, rfftfreq
from scipy.stats import skew,kurtosis

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    source on on a prise cet classe : https://github.com/Bjarten/early-stopping-pytorch
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score - self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



def frequencyConversionPerElectrode(data,fs=1024,eeg_bands={'Delta': (1, 4),
             'Theta': (4, 8),
             'Alpha': (8, 12),
             'Beta': (12, 30),
             'Gamma': (30, 45)}):
    """
    Fonction qui convertit les données qui sont en time domain en fréquence par electrode,elle prend en entrée les données eeg sours forme de matrice
    représentant l'amplitude , la fréquence d'échantillonage et les intervales de fréquences voulues
    source:
    """
    # Get real amplitudes of FFT (only in postive frequencies)
    fft_vals = abs(rfft(data))

    # Get frequencies for amplitudes in Hz
    fft_freq = rfftfreq(data.shape[1], 1.0/fs)

    # Take the mean of the fft amplitude for each EEG band
    arrC = []
    for band in eeg_bands:
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
                           (fft_freq <= eeg_bands[band][1]))[0]
        freqPerElect = np.mean(fft_vals[:,freq_ix],axis=1)

        arrC.append(freqPerElect.reshape(-1,1))

    arrM = np.concatenate(arrC,axis=1)
    return arrM

def frequencyPerEeg(data,**kwargs):

    """Fonction qui prend les fréquence par électrodes et le transforme en fréquence par eeg, donc fait la moyenne pour chaque fréquence"""
    return frequencyConversionPerElectrode(data,**kwargs).mean(axis=0).reshape(1,-1)

def featuresExtraction(data,**kwargs):

    """Fonction qui prend les données eeg extraite sous forme d'amplitude, en extrait la férquences grâces à la
    trasnformation de fourier et d'autres features simples également"""
    fourierPerEeg = frequencyPerEeg(data,**kwargs)
    eegMean = data.mean()
    eegStd = data.std()
    eegMin = data.min()
    eegMax = data.max()
    eegKurtosis = kurtosis(data, axis=None)
    eegSkewness = skew(data, axis=None)

    simpleFeatures = np.array([eegMean,eegStd,eegMin,eegMax,eegKurtosis,eegSkewness])
    finalSample = np.concatenate((fourierPerEeg,simpleFeatures.reshape(1,-1)),axis=1)

    return finalSample

def epochsToSimpleFeatures(arrayData,**kwargs):
    """Fonction qui applique la fonction précédente à tout un ensemple d'epochs et non seulemetn à une seule epoch"""
    arrFinal = [featuresExtraction(arr,**kwargs) for arr in arrayData]
    arrFinal = np.concatenate(arrFinal)
    return arrFinal

def train_test_split_sub(df,testratio=0.2,random_seed=42):
    """Function that split the data by patient, instead of randomly to avoid cross-contamination and returns train,validation and test dataset as dfs"""
    np.random.seed(random_seed)
    subSet = set(df["subId"].unique())
    testSubs = set(np.random.choice(list(subSet),int(len(subSet)*testratio),replace=False))
    trainSubs = subSet-testSubs

    dftrain = df[df["subId"].isin(trainSubs)]
    dftest = df[df["subId"].isin(testSubs)]

    return dftrain,dftest

def databykeys(hdfFiles,hdfkeys):
    """Function that return the data for the patients in the hdfkeys that are given , in the form of  a dask array"""
    DataList = [da.from_array(hdfFile[i][j]) for hdfFile in hdfFiles for i in hdfkeys for j in hdfFile[i].keys() if "data" in j and i in hdfFile.keys() ]
    data = da.concatenate(DataList, axis=0)

    LabelsList = [da.from_array(hdfFile[i][j]) for hdfFile in hdfFiles for i in hdfkeys for j in hdfFile[i].keys() if "label" in j and i in hdfFile.keys() ]
    labels = da.concatenate(LabelsList, axis=0)

    IdsList = [da.from_array(hdfFile[i][j]) for hdfFile in hdfFiles for i in hdfkeys for j in hdfFile[i].keys() if "ids" in j and i in hdfFile.keys() ]
    ids = da.concatenate(IdsList, axis=0)

    return data,labels,ids

def train_test_split(hdfFiles,dataSample=1,testratio=0.2,random_seed=True,random_seed_value =42):
    """Function that splits the time domain data of the eeg that are in the hdfFile"""
    keys = [key for hdfFile in hdfFiles for key in hdfFile.keys() ]
    listavailpts = [int(i[-3:]) for i in keys]
    numPts = int(len(keys)*dataSample)
    
    if random_seed:
        np.random.seed(random_seed_value)
        random.seed(random_seed_value)

    myset = set(np.random.choice(listavailpts,numPts,replace=False))
    print(myset)
    testIds = random.sample(myset, int(testratio*len(myset)))
    trainIds = myset - set(testIds)
    
    trainKeys = [i for i in keys if int(i[-3:]) in trainIds]
    testKeys = [i for i in keys if int(i[-3:]) in testIds]
    
    trainData = databykeys(hdfFiles,trainKeys)
    testData = databykeys(hdfFiles,testKeys)

    return trainData,testData



def create_loaders(trainData,testData, batch_size, valid=False, valid_size=0.2, random_seed=True, random_seed_value= 42):
    """Function that creates the loaders with the data given , it can return a validation loader if specified """
    dataArrTrain = EegDataSetDask(trainData)
    dataArrTest = EegDataSetDask(testData)
    
    if not valid:
        dlDataArrTrain = DataLoader(dataArrTrain, batch_size=batch_size, shuffle=True,num_workers=4,pin_memory=True)
        dlDataArrtest = DataLoader(dataArrTest, batch_size=batch_size, shuffle=True,num_workers=4,pin_memory=True)
        return dlDataArrTrain,dlDataArrtest
    
    if random_seed:
        np.random.seed(random_seed_value)
        random.seed(random_seed_value)
        
    num_train = len(dataArrTrain)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    dlDataArrTrain = DataLoader(dataArrTrain, batch_size=batch_size,sampler=train_sampler,num_workers=0,pin_memory=False)
    
    dlDataArrValid = DataLoader(dataArrTrain, batch_size=batch_size,sampler=valid_sampler,num_workers=0,pin_memory=False)
    
    dlDataArrtest = DataLoader(dataArrTest, batch_size=batch_size, shuffle=True,num_workers=0,pin_memory=False)
    
    return dlDataArrTrain, dlDataArrValid, dlDataArrtest





def train_model(model,train_loader, valid_loader, batch_size, patience, n_epochs,device,criterion,optimizer,uniqueId,typeModel="rnn",delta=0):
    """Function that trains a rnn,gru, or lstm model, with early stopping
    part of code from : https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb"""
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True,path=uniqueId,delta=0)
    
    if typeModel.lower()=="rnn":
        for epoch in range(1, n_epochs + 1):

            ###################
            # train the model #
            ###################
            model.train() # prep model for training
            for batch, data_b in enumerate(train_loader, 1):

                inputs, labels = data_b[0].to(torch.float32).to(device),data_b[1].to(torch.int64).to(device)
                # clear the gradients of all optimized variables
                for param in model.parameters():
                    param.grad = None
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)


                # Backward and optimize
                loss.backward()
                optimizer.step()
                # record training loss
                train_losses.append(loss.item())

            ######################    
            # validate the model #
            ######################
            model.eval() # prep model for evaluation
            for datav in valid_loader:
                # forward pass: compute predicted outputs by passing inputs to the model
                inputs, labels = datav[0].to(torch.float32).to(device),datav[1].to(torch.int64).to(device)
                output = model(inputs)
                # calculate the loss
                loss = criterion(output, labels)
                # record validation loss
                valid_losses.append(loss.item())

            # print training/validation statistics 
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(n_epochs))

            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')

            print(print_msg)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
    
    if typeModel.lower()=="gru":
        n_total_steps = len(train_loader)
        for epoch in range(1, n_epochs + 1):

            ###################
            # train the model #
            ###################
            model.train() # prep model for training
            h = model.init_hidden(batch_size)
            for batch, data_b in enumerate(train_loader, 1):
                #print(batch,len(train_loader))
                if batch==n_total_steps:
                    break
                inputs, labels = data_b[0].to(torch.float32).to(device),data_b[1].to(torch.int64).to(device)
                h = h.data
                # clear the gradients of all optimized variables
                for param in model.parameters():
                    param.grad = None
                
                # Forward pass
                outputs,h = model(inputs,h)
                loss = criterion(outputs, labels)


                # Backward and optimize
                loss.backward()
                optimizer.step()
                # record training loss
                train_losses.append(loss.item())

            ######################    
            # validate the model #
            ######################
            model.eval() # prep model for evaluation
            for datav in valid_loader:
                # forward pass: compute predicted outputs by passing inputs to the model
                inputs, labels = datav[0].to(torch.float32).to(device),datav[1].to(torch.int64).to(device)
                h = model.init_hidden(inputs.shape[0])
                output,h = model(inputs,h)
                # calculate the loss
                loss = criterion(output, labels)
                # record validation loss
                valid_losses.append(loss.item())

            # print training/validation statistics 
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(n_epochs))

            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')

            print(print_msg)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
    
    if typeModel.lower()=="lstm":
        n_total_steps = len(train_loader)
        for epoch in range(1, n_epochs + 1):

            ###################
            # train the model #
            ###################
            model.train() # prep model for training
            h = model.init_hidden(batch_size)
            for batch, data_b in enumerate(train_loader, 1):
                #print(batch,len(train_loader))
                if batch==n_total_steps:
                    break
                inputs, labels = data_b[0].to(torch.float32).to(device),data_b[1].to(torch.int64).to(device)
                h = tuple([e.data for e in h])
                # clear the gradients of all optimized variables
                for param in model.parameters():
                    param.grad = None
                
                # Forward pass
                outputs,h = model(inputs,h)
                loss = criterion(outputs, labels)


                # Backward and optimize
                loss.backward()
                optimizer.step()
                # record training loss
                train_losses.append(loss.item())

            ######################    
            # validate the model #
            ######################
            model.eval() # prep model for evaluation
            for datav in valid_loader:
                # forward pass: compute predicted outputs by passing inputs to the model
                inputs, labels = datav[0].to(torch.float32).to(device),datav[1].to(torch.int64).to(device)
                h = model.init_hidden(inputs.shape[0])
                output,h = model(inputs)
                # calculate the loss
                loss = criterion(output, labels)
                # record validation loss
                valid_losses.append(loss.item())

            # print training/validation statistics 
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(n_epochs))

            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')

            print(print_msg)

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
                
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(uniqueId))

    return  model, avg_train_losses, avg_valid_losses
    

def vizEarlyStopping(train_loss,valid_loss,filepath):
    """ visualize the train loss and validation loss
    part of code from : https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
    """

    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 1) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    fig.savefig(filepath+'.png', bbox_inches='tight')

def testmodel(model,test_loader,batch_size,criterion,device,file,typeModel="rnn"):
    """testing of the data for rnn,gru and lstm"""
    test_loss = 0.0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))

    model.eval() # prep model for evaluation
    
    if typeModel.lower()=="rnn":
        for data_test in test_loader:

            # forward pass: compute predicted outputs by passing inputs to the model
            inputs, labels = data_test[0].to(torch.float32).to(device),data_test[1].to(torch.int64).to(device)
            output = model(inputs)
            if len(labels.data) != batch_size:
                break
            # calculate the loss
            loss = criterion(output, labels)
            # update test loss 
            test_loss += loss.item()*inputs.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
            # calculate test accuracy for each object class
            for i in range(batch_size):
                label = labels.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        # calculate and print avg test loss
        test_loss = test_loss/len(test_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))
        file.write('Test Loss: {:.6f}\n'.format(test_loss))
        for i in range(2):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    str(i), 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
                file.write('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    str(i), 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % i)
                file.write('Test Accuracy of %5s: N/A (no training examples)' % i)

    if typeModel.lower()=="gru" or typeModel.lower()=="lstm":
        for data_test in test_loader:

            # forward pass: compute predicted outputs by passing inputs to the model
            inputs, labels = data_test[0].to(torch.float32).to(device),data_test[1].to(torch.int64).to(device)
            h = model.init_hidden(inputs.shape[0])
            output,h = model(inputs,h)
            
            if len(labels.data) != batch_size:
                break
            # calculate the loss
            loss = criterion(output, labels)
            # update test loss 
            test_loss += loss.item()*inputs.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
            # calculate test accuracy for each object class
            for i in range(batch_size):
                label = labels.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        # calculate and print avg test loss
        test_loss = test_loss/len(test_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))
        file.write('Test Loss: {:.6f}\n'.format(test_loss))
        for i in range(2):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    str(i), 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
                file.write('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    str(i), 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % i)
                file.write('Test Accuracy of %5s: N/A (no training examples)' % i)
    

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    file.write('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    return (100.0 * np.sum(class_correct) / np.sum(class_total))


class EegDataSetDask(Dataset):
    """Class of dataset for the eeg data that is a dask dataset"""
    def __init__(self,dataArray):
        #data loading
        self.x = dataArray[0]
        self.labels = dataArray[1]
        self.ids = dataArray[2]

    def __len__(self):
        # len(dataset)
        return self.x.shape[0]

    def __getitem__(self, idx):
        # if downsamplig use torch.from_numpy(self.x[idx][:,[i for i in range(10240) if i%50==0]].compute())
        return torch.from_numpy(self.x[idx].compute()),torch.from_numpy(self.labels[idx].compute())
    

# Source code for RNN, GRU, LSTM:
# Taken from: https://github.com/python-engineer/pytorch-examples/blob/master/rnn-lstm-gru/main.py

class RNN(nn.Module):
    """Rnn model with one rnn layer and on final linear layer"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes,device):
        super(RNN, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.reshape(x,(x.shape[0],x.shape[2],x.shape[1]))
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class GRUmodel(nn.Module):
    """Gru model with  with one gru layer and one final linear fully connected layer"""
    def __init__(self, input_dim, hidden_dim, n_layers,num_classes,device):
        super(GRUmodel, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        x = torch.reshape(x,(x.shape[0],x.shape[2],x.shape[1]))
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden
    
class LSTMNet(nn.Module):
    """Lstm model with on lstm layer and one final Linear layer"""
    def __init__(self, input_dim, hidden_dim,n_layers,num_classes,device):
        super(LSTMNet, self).__init__()
        self.device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        x = torch.reshape(x,(x.shape[0],x.shape[2],x.shape[1]))
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden
    
class Cnn_eeg(nn.Module):
    """On cnn model wiht 2 conv layers with maxpool, and 3 fully connected layers at the end """
    def __init__(self,batch_size):
        super().__init__()
        self.batch_size=batch_size
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(self.linear_input_neurons(), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # print("0:",x.shape)
        x = torch.reshape(x,(x.shape[0],1,x.shape[1],x.shape[2]))
        # print("1:",x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print("2:",x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print("3:",x.shape)
        x = x.view(x.shape[0], -1) # flatten all dimensions except batch
        # print("4:",x.shape)
        x = F.relu(self.fc1(x))
        # print("5:",x.shape)
        x = F.relu(self.fc2(x))
        # print("6:",x.shape)
        x = self.fc3(x)
        # print("7:",x.shape)
        return x
        # here we apply convolution operations before linear layer, and it returns the 4-dimensional size tensor.
    def size_after_relu(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        return x.size()[1:]


    # after obtaining the size in above method, we call it and multiply all elements of the returned size.
    def linear_input_neurons(self):
        size = self.size_after_relu(torch.rand(self.batch_size, 1, 72, 1024))
        m = 1
        for i in size:
            m *= i

        return int(m)
    
    
    
    
    
