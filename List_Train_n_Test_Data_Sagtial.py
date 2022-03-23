import nibabel as nib
import numpy as np
import pathlib
import matplotlib.pyplot as plt
#print('check 12')

Train_voor = []
Train_na = []

path = pathlib.Path(__file__).parent
for timestamp in ["-001h", "024h"]:
    for mouse in ["M03", "M04", "M05", "M06", "M07"]:
        if timestamp == "-001h":
            path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
            Train_voor.append(nib.load(path_ct).get_fdata())
        else: 
            path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
            Train_na.append(nib.load(path_ct).get_fdata())           


# nuttige slices lopen van idx 25 tot 125
Train_Data_001h = []
Train_Data_024h = []
for mouse in Train_voor:
    for slice in mouse[25:125]:
        Train_Data_001h.append(slice)

for mouse in Train_na:
    for slice in mouse[25:125]:
        Train_Data_024h.append(slice)


Test_Data_001h = []
Test_Data_024h = []

for timestamp in ["-001h", "024h"]:
    mouse = "M08"
    path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
    ct = nib.load(path_ct).get_fdata()
    for slice in ct[25:125]:
        if timestamp == "-001h":
            Test_Data_001h.append(slice)
        else:
            Test_Data_024h.append(slice)


#Train_Data_001h (500 slices)
#Train_Data_024h (500 slices)
#Test_Data_001h  (100 slices)
#Test_Data_024h  (100 slices)




