import nibabel as nib
import numpy as np
import pathlib
import matplotlib.pyplot as plt
#print('check 12')

# Data:
# 6 mice
# 3 time instances (CT-scan)
# 154 slices per scan; 121x242

d = {}
o = {}
path = pathlib.Path(__file__).parent
for mouse in ["M03", "M04", "M05", "M06", "M07", "M08"]:
    #print(mouse)
    for timestamp in ["-001h", "0.25h", "024h"]:
        d[timestamp] = {}
        
        path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
        path_organ = path / f"processed/{mouse}_{timestamp}_Organ280.img"
        #path_class = path / f"original/{mouse}_{timestamp}/Organ_280.cls"
        #if not path_organ.is_file():
        #    path_organ = path / f"original/{mouse}_{timestamp}/Organ1_280.img"
        #if not path_class.is_file():
        #    path_class = path / f"original/{mouse}_{timestamp}/Organ1_280.cls"



        ct = nib.load(path_ct).get_fdata()
        organ = nib.load(path_organ).get_fdata()
        d[str(mouse)+'_'+str(timestamp)] = [i for i in ct]
        o[str(mouse)+'_'+str(timestamp)+'_Organ'] = [i for i in organ]
    


# Test evolutie id tijd

# Plots
fig, axs = plt.subplots(1,3, figsize=(10,10), sharex=True)
axs[0].imshow(d["M08_-001h"][120],cmap = 'bone')
axs[1].imshow(d["M08_0.25h"][120],cmap = 'bone')
axs[2].imshow(d["M08_024h"][120],cmap = 'bone')
plt.show()

#print((d["M08_-001h"][80]))

#Testje Organ --> onduidelijk nog

# Plots
#fig, axs = plt.subplots(1,3, figsize=(10,10), sharex=True)
#axs[0].imshow(o["M08_-001h_Organ"][110],cmap = 'bone')
#axs[1].imshow(o["M08_0.25h_Organ"][110],cmap = 'bone')
#axs[2].imshow(o["M08_024h_Organ"][110],cmap = 'bone')

#plt.show()