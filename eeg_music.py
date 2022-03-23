from glob import glob
import mne
import numpy as np

print('EEG ANALYSIS: MUSIC THINKING \n')

all_eeg_path = glob('Music Thinking/sub-*/ses-*/eeg/*.set')
print(f'Number of individuals: {len(all_eeg_path)}')

# Array for the labels
behaviour_data = np.genfromtxt('Music Thinking/stimuli/Behavioural_data.txt')
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


# Function that reads EEG Data (set extension file)
def read_set_data(path):
    music_data = mne.io.read_raw_eeglab(path, preload=True)
    # Preprocessing (Filtering)
    # music_data.filter(l_freq = 0.1, h_freq = 60)

    epochs = mne.make_fixed_length_epochs(music_data, duration=4, overlap=1)
    music_array = epochs.get_data()

    print(f'Dimensions of the tensor: {music_array.shape}')
    return music_array


# Plot visualization
ex = mne.io.read_raw_eeglab(all_eeg_path[0], preload = True)
ex.plot()
ex.plot_psd()

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
