from glob import glob
import os
import mne
import numpy as np
import sys
import random

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

for i in range(30):
    print('all_eeg_path =',all_eeg_path[i])
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
like_path = like_path[0:40]
dislike_path = dislike_path[0:40]

print('shape of like and dislike path : ',like_path.shape[0], dislike_path.shape[0]) 

print('----------------')
# Function that reads EEG Data (set extension file)
def read_set_data(path):
    print('---- File path -----')
    print('Begin read with path = ',path)
    music_data = mne.io.read_raw_eeglab(path, preload=True)
    
    # Preprocessing (Filtering)
    # music_data.filter(l_freq = 0.1, h_freq = 60)
    #import pdb; pdb.set_trace()
    
    epochs = mne.make_fixed_length_epochs(music_data, duration=3, overlap=2,preload = True)
    music_array = epochs.get_data()
    #shape music_array = 134x129x750
    
    music_array = music_array[:,:,:750]
    #shape music_array = 134x129x750
    
    #keep only 4 epochs for training ?
    number_epochs = 4
    array_epochs = np.empty(number_epochs, dtype = object)

    for i in range(number_epochs):
        chosen_number = random.randint(0, music_array.shape[0]-1)
        #print(chosen_number)
        array_epochs[i] = music_array[chosen_number]
        
    #print(f'Dimensions of the tensor: {array_epochs[0].shape}')
    
    #shape array_epochs : 129x750
    #129 electrodes x 750 time points
    return array_epochs


#------- read all eeg files ------ #
print('------ Step read all eeg files --------')

music_array_1 = read_set_data(dislike_path[20])
print('music_array_1 shape =',np.shape(music_array_1))

#music_array_2 = read_set_data(like_path[1])
#print('music_array_2 shape =',np.shape(music_array_2))
'''
random.seed(0)

like_epoch_array = np.empty((len(like_path)), dtype = object)
dislike_epoch_array = np.empty((len(dislike_path)), dtype = object)
print('shape of like and dislike epoch array : ',like_epoch_array.shape[0], dislike_epoch_array.shape[0])

for i in range(len(like_path)):
    print('counting i = ',i)
    like_epoch_array[i] = read_set_data(like_path[i])
    
for i in range(len(dislike_path)):
    print('counting i = ',i)
    dislike_epoch_array[i] = read_set_data(dislike_path[i])

    
print('shape of like and dislike epoch array : ',like_epoch_array.shape[0], dislike_epoch_array.shape[0])
#-----------
'''

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
