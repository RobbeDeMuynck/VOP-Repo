import nibabel as nib
import numpy as np
import pathlib
import matplotlib.pyplot as plt
#print('check 12')

Train_voor = []
Train_na = []

path = pathlib.Path(__file__).parent
for mouse in ["M03", "M04", "M05", "M06", "M07", "M08"]:
    for timestamp in ["-001h", "024h"]:
        if timestamp == "-001h":
            path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
            Train_voor.append(nib.load(path_ct).get_fdata())
        else:
            path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
            Train_na.append(nib.load(path_ct).get_fdata())


# Plots
fig, axs = plt.subplots(1,2, figsize=(10,10), sharex=True)
axs[0].imshow(Train_voor[0][10],cmap = 'bone')
axs[1].imshow(Train_na[5][10],cmap = 'bone')

plt.show()