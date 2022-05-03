import h5py
from tqdm import tqdm
import mne
import os
import numpy as np
from Utilities import epochsToSimpleFeatures
import pandas as pd

#titres des fichiers avec les tâches pour les patients
tasks = ["task-med1breath_eeg","task-med2_eeg","task-think1_eeg","task-think2_eeg"]

#Dossier avec les données Eeg
eegDir = "W:\RnnMachineLearning\eegData"

#fichier hdf de destination pour les données extraites
h5file = "W:\RnnMachineLearning\meditationDataFourierSimpleFeaturesGExtended.hdf5"
f = h5py.File(h5file, "w")


#lecture de chaque fichier eeg puis extraction avec la librairie mne avant de les enregistrer dans un fichier hdf
for i in tqdm(range(1,99)):
    try:
        lisArraysData = []
        if i<10:
            sub = "00"+str(i)
        else:
            sub= "0"+str(i)

        for task in tqdm(tasks):
            fileName = f"sub-{sub}_{task}.bdf"
            medState = "med" in fileName
            print("File being read:",fileName)
            ex = mne.io.read_raw_bdf(os.path.join(eegDir,fileName),preload=True)

            #on drop les channels excedentaires , ceux pour les pupiles et mastoides et ceux non eeg
            if ex.info.get("nchan")==80:
                ex.drop_channels(('EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8',
                                  'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp', 'Status'))
            else:
                ex.drop_channels(('EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8','Status'))
            Epochs = mne.make_fixed_length_epochs(ex,duration=10,verbose=False)
            arrEpochs = Epochs.get_data()

            #Certains fichiers ont plus de timepoints on ne sait pas pourquoi on les drop donc
            if arrEpochs.shape[2]==20480:
                print("Wrong size:",20480)
                continue

            #Extraction avec fourier des puissances correspondantes aux ondes ,delta,theta,alpha,beta et gamma

            features = epochsToSimpleFeatures(arrEpochs,fs=1024,eeg_bands={'Delta': (1, 4),
             'Theta': (4, 8),
             'Alpha': (8, 12),
             'Beta': (12, 30),
             'Gamma': (30, 110)})
            featuresIdsMed = np.zeros((features.shape[0],features.shape[1]+2))
            featuresIdsMed[:,:-2] = features
            featuresIdsMed[:,-2]=i
            featuresIdsMed[:,-1]=medState

            lisArraysData.append(featuresIdsMed)

        #sauvegarde dans le fichier hdf sous le nom du patient
        f.create_dataset(f"sub-{sub}",data=np.concatenate(lisArraysData,axis=0))

    except:
        continue
f.close()