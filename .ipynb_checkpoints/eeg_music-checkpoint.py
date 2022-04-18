'''
Authors : Equipe EEG
Fichier pour training le model RNN pour dataset Music_eeg
Last updated : 15-04-202
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

    music_data = mne.io.read_raw_eeglab(path, preload=False)
    
    channels = (129 * ['E'])
    for i in range(1, 130):
        channels[i - 1] += str(i)
        
    # Select the 64 electrodes among the 129 electrodes
    number_electrodes = 64
    kept_channels = np.empty((number_electrodes), dtype = object)
    index_channels = np.zeros((number_electrodes), dtype = int)
    
    for i in range(len(kept_channels)):
        kept_channels[i] = channels[2*i]
        index_channels[i] = 2*i
    
    dropped_channels = np.delete(channels, index_channels)
    music_data.drop_channels(dropped_channels)
    
    epochs = mne.make_fixed_length_epochs(music_data, duration=3, overlap=1,preload = False)
    
    music_array = epochs.get_data()
    music_array = music_array[:,:,:750]
    
    number_epochs = 20
    array_epochs = np.empty(number_epochs, dtype = object)

    random.seed(0)
    for i in range(number_epochs):
        chosen_number = random.randint(0, music_array.shape[0]-1)
        
        array_epochs[i] = music_array[chosen_number]                #64x750
        #array_epochs[i] = music_array[20+i]
        
    print(f'Dimensions of the tensor: {array_epochs[0].shape}')
    
    #64 electrodes x 750 time points
    return array_epochs

#fonction principale pour lire les fichiers eeg et pretraiter
def music_preprocessing(path) :
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

    #test 
    like_path = like_path[0:35]
    dislike_path = dislike_path[0:36]
    print('shape of like and dislike path : ',like_path.shape[0], dislike_path.shape[0]) 

    print('----------------')



    #------- read all eeg files ------ #
    print('------ Step read all eeg files --------')

    ## NOTE : dislike_path[20] error file, dislike_path[33]?

    #music_array_1 = read_set_data(dislike_path[33])
    #print('music_array_1 shape =',np.shape(music_array_1))

    #music_array_2 = read_set_data(like_path[1])
    #print('music_array_2 shape =',np.shape(music_array_2))

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

    X = np.hstack(np.append(like_epoch_array,dislike_epoch_array)) #15*4 = 60 *2 = 120
    Y = np.hstack(np.append(like_epoch_labels,dislike_epoch_labels))
    print(np.shape(X),np.shape(Y))
    
    return X,Y

