import nibabel as nib
import numpy as np
import pathlib
import matplotlib.pyplot as plt


d = {}
path = pathlib.Path(__file__).parent
for mouse in ["M03", "M04", "M05", "M06", "M07", "M08"]:
    #print(mouse)
    for timestamp in ["-001h", "024h"]:
        d[timestamp] = {}
        
        path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
        ct = nib.load(path_ct).get_fdata()

        d[str(mouse)+'_'+str(timestamp)] = [ct[:,:,i] for i in range(ct.shape[-1])]
    
# Test evolutie id tijd

# Plots
# fig, axs = plt.subplots(1,2, figsize=(10,10), sharex=True)
# axs[0].imshow(d["M08_-001h"][120],cmap = 'bone')
# axs[1].imshow(d["M08_024h"][120],cmap = 'bone')
# plt.show()

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


Train_Data_001h = []
Train_Data_024h = []

for mouse in Train_voor:
    for i in range(mouse.shape[-1]):
        Train_Data_001h.append(mouse[:,:,i])


for mouse in Train_na:
    for i in range(mouse.shape[-1]):
         Train_Data_024h.append(mouse[:,:,i])


Test_Data_001h = []
Test_Data_024h = []

for timestamp in ["-001h", "024h"]:
     mouse = "M08"
     path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
     ct = nib.load(path_ct).get_fdata()
     for i in range(ct.shape[-1]):
        if timestamp == "-001h":
            Test_Data_001h.append(ct[:,:,i])
        else:
            Test_Data_024h.append(ct[:,:,i])

#Train_Data_001h (1210 slices)
#Train_Data_024h (1210 slices)
#Test_Data_001h  (242 slices)
#Test_Data_024h  (242 slices)




