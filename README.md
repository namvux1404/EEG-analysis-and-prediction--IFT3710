# IFT3710

TITLE: Analyzing the performance of transfer learning with different EEG datasets

• First Dataset: Meditation vs thinking task

o https://openneuro.org/datasets/ds003969/versions/1.0.0

• Second Dataset: Music Listening

o https://openneuro.org/datasets/ds003774/versions/1.0.0

Team Members
- Ronnie Liu: ronnie.liu@umontreal.ca
- Anas Bourhim: anas.bourhim@umontreal.ca
- Hichem Sahraoui: hichem.sahraoui@umontreal.ca
- Van Nam Vu: van.nam.vu@umontreal.ca

Description of the Project
- Motivation:
  - Generally, in machine learning, every time there’s a different dataset, we need to
create a model that minimizes the error of validation (or test). However, with two
different EEG datasets, it can be challenging when it comes to accurately train
models in neuroscience, especially with deep learning methods. This is because
many experiments don’t have enough data due to the high complexity and cost of
the procedures for data gathering. Furthermore, preprocessing EEG data can lead
to very complex datasets (a lot of attributes for a few datapoints). We decide to
introduce the concept of transfer learning to approve the accuracy of training a
model with few EEG datapoints.

- Objectives:
  - Being able to preprocess and extract essential features for different EEG datasets
  - Understand the concept of transfer learning with different EEG datasets
  - Compare the accuracy of the dataset of music thinking between different methods
  (data training from scratch vs transfer learning)

**************************
STEPS FOR RUNNING THE CODE
**************************

RNN MODEL AND TRANSFER LEARNING

PART I: TRANSFER LEARNING BETWEEN DATASETS (MUSIC GROUP 1 - MEDITATION)
- Step 1: Train the RNN Model for the music dataset (Group 1)
	-> python IFT3710/train_music.py /home/liuronni/projects/def-sponsor00/datasets/EEG/Music_eeg_raw 1
- Step 2: Train the RNN Model for the meditation dataset
	-> python IFT3710/train_meditation.py /home/liuronni/projects/def-sponsor00/datasets/EEG/Med_eeg_raw
- Step 3: Apply Transfer Learning from music to meditation datasets
	-> python IFT3710/transfer_learning.py 1

PART II: TRANSFER LEARNING IN ONE DATASET (MUSIC GROUP 1 - MUSIC GROUP 2)
- Step 1: Train the RNN Model for the music dataset (Group 1)
	-> python IFT3710/train_music.py /home/liuronni/projects/def-sponsor00/datasets/EEG/Music_eeg_raw 1
- Step 2: Train the RNN Model for the music dataset (Group 2)
	-> python IFT3710/train_music.py /home/liuronni/projects/def-sponsor00/datasets/EEG/Music_eeg_raw 2
- Step 3: Apply Transfer Learning from group 1 to group 2
	-> python IFT3710/transfer_learning.py 2

Xgboost

	1-Upsate the path for the data on the participants and for the data of the daframes for the features Extracted from the data in the Xgboost notebook
	2-Run the Xgboost notebooks, you can change the hyperparameters for the model optimisation


