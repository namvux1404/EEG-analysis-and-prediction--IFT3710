import h5py
from tqdm import tqdm
tasks = ["task-med1breath_eeg",
"task-med2_eeg",
"task-think1_eeg",
"task-think2_eeg"]

eegDir = "D:\RnnMachineLearning\eegData"
h5file = "D:\RnnMachineLearning\meditationData.hdf5"
f = h5py.File(h5file, "r+")



for i in tqdm(range(6,99)):
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

        if ex.info.get("nchan")==80:
            ex.drop_channels(('GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp', 'Status'))
        else:
            ex.drop_channels(('Status'))
        Epochs = mne.make_fixed_length_epochs(ex,duration=10,verbose=False)
        arrEpochs = Epochs.get_data()
        print(arrEpochs.shape)
        if arrEpochs.shape[2]==20480:
            print("Wrong size:",20480)
            continue

        for arr in arrEpochs:
            listData.append(arr)
            listLabels.append(medState)
            listIds.append(i)
            # listData.append([sub,i,medState])
    #print(np.array(listData)[:,1].shape)

    f.create_dataset(f"sub-{sub}/sub-{sub}-data",data=np.array(listData))
    f.create_dataset(f"sub-{sub}/sub-{sub}-labels",data=np.array(listLabels))
    f.create_dataset(f"sub-{sub}/sub-{sub}-ids",data=np.array(listIds))
   # da.to_npy_stack(f"D:/RnnMachineLearning/stack2/sub-{sub}/",da.from_array(np.array(listData,dtype="object"),chunks=(len(listData))), axis=0)
    # np.save(f"D:\RnnMachineLearning\stackData\sub-{sub}.npy",np.array(listData))