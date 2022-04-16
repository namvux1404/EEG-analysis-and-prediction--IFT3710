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