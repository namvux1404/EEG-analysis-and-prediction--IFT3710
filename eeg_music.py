'''
Authors: EEG Team
File: eeg_music.py
- This file focuses on reading the raw EEG data from the music dataset.
- The extension files of EEG signals are SET (eeglab).
- This file will also preprocessing the dataset and separate them into features and labels.
'''

from glob import glob
import os
import mne
import numpy as np
import sys
import random
from tqdm.auto import tqdm

# Function that reads EEG Data (set extension file)
def read_set_data(path):
    print('---- File path -----')
    print('Begin read with path = ',path)

    # Read the raw EEGLAB datafile
    music_data = mne.io.read_raw_eeglab(path, preload=False)
    
    # Naming the electrodes
    channels = (129 * ['E'])
    for i in range(1, 130):
        channels[i - 1] += str(i)
        
    # Select the 64 electrodes among the 129 electrodes
    number_electrodes = 64
    kept_channels = np.empty((number_electrodes), dtype = object)
    index_channels = np.zeros((number_electrodes), dtype = int)
    
    # The channels that are kept will be 0, 2, 4, ..., 126.
    # The mapping of the electrodes aren't accurate compared to the ones 
    # in the meditation dataset, but this naive method ensures that
    # the electrodes that are kept cover most of the individual's head.
    for i in range(len(kept_channels)):
        kept_channels[i] = channels[2*i]
        index_channels[i] = 2*i
    
    dropped_channels = np.delete(channels, index_channels)
    music_data.drop_channels(dropped_channels)
    
    # Separation of each individual's EEG signals into epochs
    # Each epoch contains 3 seconds of signals with an overlap of 1 second.
    epochs = mne.make_fixed_length_epochs(music_data, duration=3, overlap=1,preload = False)
    
    # Convert the epochs into an numpy array type
    music_array = epochs.get_data()
    
    # Some epoch contains more then 750 timepoints in order to represent 3 seconds of signals.
    # In order to prevent 
    music_array = music_array[:,:,:750]
    
    # For each individual, we will select 20 epochs randomly
    number_epochs = 20
    array_epochs = np.empty(number_epochs, dtype = object)

    random.seed(0)
    for i in range(number_epochs):
        chosen_number = random.randint(0, music_array.shape[0]-1)
        
        array_epochs[i] = music_array[chosen_number]               
        
    print(f'Dimensions of the tensor: {array_epochs[0].shape}')
    
    # For each epoch, we will have 64 electrodes x 750 timepoints
    return array_epochs



# Principal function that reads all individual's EEG signals and preprocess them
# begin and end: start and ending index in the preprocessing
def music_preprocessing(path, begin, end) :
    print('--Fichier preprocessing-----')
    print('path =',path)

    all_eeg_path = sorted(glob(path + '/*.set')) 

    print(f'Number of individuals: {len(all_eeg_path)}')

    behaviour_data = np.genfromtxt(path + '/stimuli_Behavioural_data.txt')
    behaviour_data = behaviour_data[1:]

    labels = behaviour_data[:, 2]
    for i in tqdm(range(labels.shape[0])):

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

    # We will only take individuals from indexes "begin" to "end"
    # for each type of labels
    like_path = like_path[begin:end]
    
    # We changed the ending index to "end + 1" since one of the individual's EEG signals aren't 
    # working (the file is not working) -> individual 22 specifically.
    dislike_path = dislike_path[begin:end + 1]
    print('shape of like and dislike path : ',like_path.shape[0], dislike_path.shape[0]) 

    print('----------------')

    #------- Read all eeg files ------ #
    print('------ Step read all eeg files --------')

    like_epoch_array = np.empty((len(like_path)), dtype = object)
    dislike_epoch_array = np.empty((len(dislike_path)), dtype = object)

    print('shape of like and dislike epoch array : ',np.shape(like_epoch_array), np.shape(dislike_epoch_array))

    random.seed(0)
    for i in tqdm(range(len(like_path))):
        print('counting i = ',i)
        like_epoch_array[i] = read_set_data(like_path[i])

    random.seed(0)
    for i in tqdm(range(len(dislike_path))):
        if (i == 20) : continue
        else:
            print('counting i = ',i)
            dislike_epoch_array[i] = read_set_data(dislike_path[i])

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

    X = np.hstack(np.append(like_epoch_array,dislike_epoch_array)) 
    Y = np.hstack(np.append(like_epoch_labels,dislike_epoch_labels))
    print(np.shape(X),np.shape(Y))
    
    return X,Y

