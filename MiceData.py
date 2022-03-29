import pathlib
import nibabel as nib

##################################### LOADING DATA  ##################################
Train_voor = []
Train_na = []

path = pathlib.Path('processed').parent
for timestamp in ["-001h", "024h"]:
     for mouse in ["M03", "M04", "M05", "M06", "M07"]:
         if timestamp == "-001h":
             path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
             Train_voor.append(nib.load(path_ct).get_fdata())
         else: 
             path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
             Train_na.append(nib.load(path_ct).get_fdata())           


Train_transversal_001h = []
Train_transversal_024h = []

for mouse in Train_voor:
    for i in range(mouse.shape[-1]):
        Train_transversal_001h.append(mouse[:,:,i])

for mouse in Train_na:
    for i in range(mouse.shape[-1]):
         Train_transversal_024h.append(mouse[:,:,i])


Test_transversal_001h = []
Test_transversal_024h = []

for timestamp in ["-001h", "024h"]:
     mouse = "M08"
     path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
     ct = nib.load(path_ct).get_fdata()
     for i in range(ct.shape[-1]):
        if timestamp == "-001h":
            Test_transversal_001h.append(ct[:,:,i])
        else:
            Test_transversal_024h.append(ct[:,:,i])

# Train_transversal_001h (1210 slices)
# Train_transversal_024h (1210 slices)
# Test_transversal_001h  (242 slices)
# Test_transversal_024h  (242 slices)