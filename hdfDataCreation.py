import h5py
from tqdm import tqdm
import mne
import os
import numpy as np

#titres des fichiers eeg des patients
tasks = ["task-med1breath_eeg","task-med2_eeg","task-think1_eeg","task-think2_eeg"]

#dossiers contenant les fichiers eeg
eegDir = "W:\RnnMachineLearning\eegData"

#fichier hdf de destination des matrices
h5file = "W:\RnnMachineLearning\meditationDataFourierFeatures.hdf5"
f = h5py.File(h5file, "w")


#Boucle on on lit les données eeg pour les 98 participants, en extrait la matrice de donnnée puis la sauvegarde dans le fichier hdf

for i in tqdm(1,99):
    try:
        listData =[]
        listLabels = []
        listIds = []
        if i<10:
            sub = "00"+str(i)
        else:
            sub= "0"+str(i)

        for task in tqdm(tasks):
            fileName = f"sub-{sub}_{task}.bdf"
            medState = "med" in fileName
            print("File being read:",fileName)
            ex = mne.io.read_raw_bdf(os.path.join(eegDir,fileName),preload=True)

            #ON enleve les channels qui sont en trop dans certains eeg, et qui ne correspondent pas à des ondes crébrales également
            if ex.info.get("nchan")==80:
                ex.drop_channels(('GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp', 'Status'))
            else:
                ex.drop_channels(('Status'))
            #on sépare en epochs avant d,en extraire la matrice
            Epochs = mne.make_fixed_length_epochs(ex,duration=10,verbose=False)
            arrEpochs = Epochs.get_data()

            #On ignore ceux dont la tailler a une anomalie au niveau de la taille , pour 10 secondes
            if arrEpochs.shape[2]==20480:
                print("Wrong size:",20480)
                continue

            #On remplis les listes des données qu'on va enregistrer dans le fichier hdf
            for arr in arrEpochs:
                listData.append(arr)
                listLabels.append(medState)
                listIds.append(i)

        #Enregistrement dans le fichier hdf des données
        f.create_dataset(f"sub-{sub}/sub-{sub}-data",data=np.array(listData))
        f.create_dataset(f"sub-{sub}/sub-{sub}-labels",data=np.array(listLabels))
        f.create_dataset(f"sub-{sub}/sub-{sub}-ids",data=np.array(listIds))

    except:
        continue
f.close()