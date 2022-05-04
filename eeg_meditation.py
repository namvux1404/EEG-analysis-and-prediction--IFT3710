'''
Authors: EEG Team
File: eeg_meditation.py
- This file focuses on reading the raw EEG data from the meditation-thinking dataset.
- The extension files of EEG signals are BDF (eeglab).
- This file will also preprocessing the dataset and separate them into features and labels.

The code features extraction is inspire from : https://www.youtube.com/watch?v=cuEV-eB3Dyo&list=PLtGXgNsNHqPTgP9wyR8pmy2EuM2ZGHU5Z&index=2
'''

from glob import glob
import mne
import numpy as np
import random
from tqdm.auto import tqdm
from scipy import stats


def read_bdf_data(path):  # Function that reads EEG Data (set extension file)
    print('---- File path -----')
    print('Begin read with path = ', path)

    # Read the raw EEGLAB datafile
    med_data = mne.io.read_raw_bdf(path, preload=True)

    # Drop the 65e electrode
    if len(med_data.ch_names) > 64:
        med_data.drop_channels(med_data.ch_names[64:])

    # Filter frequency from 4 to 45
    med_data.filter(l_freq=4, h_freq=45)

    # Separation of each individual's EEG signals into epochs
    # Each epoch contains 0.75 seconds of signals with an overlap of 0.25 second.
    epochs = mne.make_fixed_length_epochs(
        med_data, duration=0.75, overlap=0.25, preload=False)

    # Convert the epochs into an numpy array type
    med_array = epochs.get_data()
    med_array = med_array[:, :, :750]

    # In music, we have number_epochs = 20.
    # In meditation, since the interval is shorter, we can add more epochs.
    number_epochs = 40
    array_epochs = np.empty(number_epochs, dtype=object)

    random.seed(0)
    for i in range(number_epochs):
        chosen_number = random.randint(0, med_array.shape[0] - 1)

        array_epochs[i] = med_array[chosen_number]  # 64x750

    print(f'Dimensions of the tensor: {array_epochs[0].shape}')

    # For each epoch, we will have 64 electrodes x 750 timepoints
    return array_epochs


# ---- Functions for features extractions ------

# feature mean
def mean(x):
    return np.mean(x, axis=-1)


def std(x):  # feature standard deviation
    return np.std(x, axis=-1)


def var(x):  # feature variance
    return np.var(x, axis=-1)


def minim(x):  # feature min
    return np.min(x, axis=-1)


def maxim(x):  # feature max
    return np.max(x, axis=-1)


def argminim(x):  # feature argmin
    return np.argmin(x, axis=-1)


def argmaxim(x):  # feature argmax
    return np.argmax(x, axis=-1)


def rms(x):  # feature root of mean square
    return np.sqrt(np.mean(x**2, axis=-1))


def abs_diff_signals(x):  # feature absolute of difference
    return np.sum(np.abs(np.diff(x, axis=-1)), axis=-1)


def skewness(x):  # feature skewness
    return stats.skew(x, axis=-1)


def kurtosis(x):  # feature kurtosis
    return stats.kurtosis(x, axis=-1)


def concatenate_features(x):  # function to concatenate all features together
    return np.concatenate((mean(x), std(x), var(x), minim(x), maxim(x), argminim(x), argmaxim(x),
                           rms(x), abs_diff_signals(x), skewness(x), kurtosis(x)), axis=-1)
# -----------------------------------------------------------


# Principal function that reads all individual's EEG signals and preprocess them
# for the size of the dataset to train, we keep only 10 first patients meditation and thinking
def med_preprocessing(path):
    print('--Fichier preprocessing-----')
    print('path =', path)

    med_path = sorted(glob(path + '/sub-0*_task-med1breath_eeg.bdf'))
    think_path = sorted(glob(path + '/sub-0*_task-think1_eeg.bdf'))

    # The proportion of people liking/disliking the music is relatively the same.
    print(f'Number of people who did meditation: {len(med_path)}')  # MEDITATE

    # DID NOT MEDITATE
    print(f'Number of people who was thinking: {len(think_path)}')
    print('----------------')

    # We onlt take 20 patients instead 50 in each category.
    med_path = med_path[0:10]
    think_path = think_path[0:11]  # there is 10th file is damage
    print('shape of med and think path : ', len(med_path), len(think_path))

    print('----------------')

    # ------- read all eeg files ------ #
    print('------ Step read all eeg files --------')

    med_epoch_array = np.empty((len(med_path)), dtype=object)
    think_epoch_array = np.empty((len(think_path)), dtype=object)

    print('shape of meditation and thinking epoch array : ',
          np.shape(med_epoch_array), np.shape(think_epoch_array))

    # read all file of meditation
    random.seed(0)
    for i in tqdm(range(len(med_path))):
        print('counting i = ', i)
        med_epoch_array[i] = read_bdf_data(med_path[i])

    # read all file of thinking
    random.seed(0)
    for i in tqdm(range(len(think_path))):
        if (i == 10):
            continue
        else:
            print('counting i = ', i)
            think_epoch_array[i] = read_bdf_data(think_path[i])

    think_epoch_array = np.delete(
        think_epoch_array, 10, 0)  # remove file damage

    print('\n')
    print('shape of med_epoch_array and med_epoch_array[0]: ', np.shape(med_epoch_array),
          np.shape(med_epoch_array[0][1]))
    print('shape of think_epoch_array and think_epoch_array[0]: ', np.shape(think_epoch_array),
          np.shape(think_epoch_array[0][1]))
    print('\n')
    # -----------

    print(' --- Create label array ---- ')
    # Assign the labels for each epoch
    med_epoch_labels = np.empty((len(med_epoch_array)), dtype=object)
    think_epoch_labels = np.empty((len(think_epoch_array)), dtype=object)

    for i in range(len(med_epoch_array)):
        med_epoch_labels[i] = len(med_epoch_array[i]) * [1]

    for i in range(len(think_epoch_array)):
        think_epoch_labels[i] = len(think_epoch_array[i]) * [0]

    print('shape med_epoch_labels and think_epoch_labels = ', np.shape(med_epoch_labels),
          np.shape(think_epoch_labels))

    print('\n')

    # concatenant array of meditation and array of thinking
    X = np.append(med_epoch_array, think_epoch_array)

    # create groupe for each patient for model Logistic Regression
    group_list = [[i]*len(j) for i, j in enumerate(X)]

    # convert to numpy array
    X = np.hstack(X)
    Y = np.hstack(np.append(med_epoch_labels, think_epoch_labels))
    group_array = np.hstack(group_list)

    # list of features for each epoch
    features = []
    for data in X:
        features.append(concatenate_features(data))

    # convert to array numpy
    features_array = np.array(features)

    print("Shape X = ", np.shape(X))
    print("Shape Y = ", np.shape(Y))
    print("Shape features = ", np.shape(features_array))
    print("Shape group = ", np.shape(group_array))

    return X, Y, features_array, group_array
