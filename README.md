# IFT3710
Github pour projet EEG - ML
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
• Motivation:
o Generally, in machine learning, every time there’s a different dataset, we need to
create a model that minimizes the error of validation (or test). However, with two
different EEG datasets, it can be challenging when it comes to accurately train
models in neuroscience, especially with deep learning methods. This is because
many experiments don’t have enough data due to the high complexity and cost of
the procedures for data gathering. Furthermore, preprocessing EEG data can lead
to very complex datasets (a lot of attributes for a few datapoints). We decide to
introduce the concept of transfer learning to approve the accuracy of training a
model with few EEG datapoints.

• Objectives:
o Being able to preprocess and extract essential features for different EEG datasets
o Understand the concept of transfer learning with different EEG datasets
o Compare the accuracy of the dataset of music thinking between different methods
(data training from scratch vs transfer learning)
