'''
Authors : Equipe EEG
Fichier pour training le model RNN pour dataset med_eeg
Last updated : 15-04-2022
'''
from glob import glob
import mne
import numpy as np
import random
from tqdm.auto import tqdm

# Function that reads EEG Data (set extension file)
def read_bdf_data(path):
    print('---- File path -----')
    print('Begin read with path = ', path)

    med_data = mne.io.read_raw_bdf(path, preload=True)

    if len(med_data.ch_names) > 64:
        med_data.drop_channels(med_data.ch_names[64:])

    med_data.filter(l_freq = 4, h_freq = 45)

    epochs = mne.make_fixed_length_epochs(med_data, duration=0.75, overlap=0.25, preload=False)

    med_array = epochs.get_data()
    med_array = med_array[:, :, :750]

    # In music, we have number_epochs = 20.
    # In meditation, since the interval is shorter, we can add more epochs.
    number_epochs = 60
    array_epochs = np.empty(number_epochs, dtype=object)

    random.seed(0)
    for i in range(number_epochs):
        chosen_number = random.randint(0, med_array.shape[0] - 1)

        array_epochs[i] = med_array[chosen_number]  # 64x750

    print(f'Dimensions of the tensor: {array_epochs[0].shape}')

    # 64 electrodes x 750 time points
    return array_epochs


# fonction principale pour lire les fichiers eeg et pretraiter
def preprocessing(path):
    print('--Fichier preprocessing-----')
    print('path =', path)

    med_path = sorted(glob(path + '/sub-0*_task-med1breath_eeg.bdf'))
    think_path = sorted(glob(path + '/sub-0*_task-think1_eeg.bdf'))

    # The proportion of people liking/disliking the music is relatively the same.
    print(f'Number of people who did meditation: {len(med_path)}')  # MEDITATE
    print(f'Number of people who was thinking: {len(think_path)}')  # DID NOT MEDITATE
    print('----------------')

    # We onlt take 20 patients instead 50 in each category.
    med_path = med_path[0:20]
    think_path = think_path[0:20]
    print('shape of med and think path : ', med_path.shape[0], think_path.shape[0])

    print('----------------')

    # ------- read all eeg files ------ #
    print('------ Step read all eeg files --------')

    med_epoch_array = np.empty((len(med_path)), dtype=object)
    think_epoch_array = np.empty((len(think_path)), dtype=object)

    print('shape of meditation and thinking epoch array : ', np.shape(med_epoch_array), np.shape(think_epoch_array))

    random.seed(0)
    for i in tqdm(range(len(med_path))):
        print('counting i = ', i)
        med_epoch_array[i] = read_bdf_data(med_path[i])

    random.seed(0)
    for i in tqdm(range(len(think_path))):
        #if (i == 20):
        #    continue
        #else:
        print('counting i = ', i)
        think_epoch_array[i] = read_bdf_data(think_path[i])

    think_epoch_array = np.delete(think_epoch_array, 20, 0)

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

    X = np.hstack(np.append(med_epoch_array, think_epoch_array))  # 15*4 = 60 *2 = 120
    Y = np.hstack(np.append(med_epoch_labels, think_epoch_labels))
    print(np.shape(X), np.shape(Y))

    return X, Y
