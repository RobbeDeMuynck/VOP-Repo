import pathlib
import nibabel as nib

##################################### LOADING DATA TRANSVERSAL  ##################################

def prep_data(test_mouse=0): #test mouse kan index 0-5 hebben
    Train_voor = []
    Train_na = []
    mouses = ["M03", "M04", "M05", "M06", "M07","M08"]
    train_mouses = [mouses[i] for i in range(len(mouses)) if i!= test_mouse]
    test_muisje = mouses[test_mouse]
    path = pathlib.Path('processed').parent
    for timestamp in ["-001h", "024h"]:
        for mouse in train_mouses:
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
        mouse = test_muisje
        path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
        ct = nib.load(path_ct).get_fdata()
        for i in range(ct.shape[-1]):
            if timestamp == "-001h":
                Test_transversal_001h.append(ct[:,:,i])
            else:
                Test_transversal_024h.append(ct[:,:,i])
    return Train_transversal_001h,Train_transversal_024h,Test_transversal_001h,Test_transversal_024h

# Train_transversal_001h (1210 slices)
# Train_transversal_024h (1210 slices)
# Test_transversal_001h  (242 slices)
# Test_transversal_024h  (242 slices)

##################################### LOADING DATA SAGITTAL ##################################
# Train_voor = []
# Train_na = []

# path = pathlib.Path(__file__).parent
# for timestamp in ["-001h", "024h"]:
#     for mouse in ["M03", "M04", "M05", "M06", "M07"]:
#         if timestamp == "-001h":
#             path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
#             Train_voor.append(nib.load(path_ct).get_fdata())
#         else: 
#             path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
#             Train_na.append(nib.load(path_ct).get_fdata())           


# # nuttige slices lopen van idx 25 tot 125
# Train_sagittal_001h = []
# Train_sagittal_024h = []
# for mouse in Train_voor:
#     for slice in mouse[25:125]:
#         Train_sagittal_001h.append(slice)

# for mouse in Train_na:
#     for slice in mouse[25:125]:
#         Train_sagittal_024h.append(slice)


# Test_sagittal_001h = []
# Test_sagittal_024h = []

# for timestamp in ["-001h", "024h"]:
#     mouse = "M08"
#     path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
#     ct = nib.load(path_ct).get_fdata()
#     for slice in ct[25:125]:
#         if timestamp == "-001h":
#             Test_sagittal_001h.append(slice)
#         else:
#             Test_sagittal_024h.append(slice)


#Train_sagittal_001h (500 slices)
#Train_sagittal_024h (500 slices)
#Test_sagittal_001h  (100 slices)
#Test_sagittal_024h  (100 slices)


##################################### LOADING DATA CORONAL ##################################
# Train_voor = []
# Train_na = []

# path = pathlib.Path('processed').parent
# for timestamp in ["-001h", "024h"]:
#      for mouse in ["M03", "M04", "M05", "M06", "M07"]:
#          if timestamp == "-001h":
#              path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
#              Train_voor.append(nib.load(path_ct).get_fdata())
#          else: 
#              path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
#              Train_na.append(nib.load(path_ct).get_fdata())           


# Train_coronal_001h = []
# Train_coronal_024h = []

# for mouse in Train_voor:
#     for i in range(mouse.shape[1]):
#         Train_coronal_001h.append(mouse[:,i,:])

# for mouse in Train_na:
#     for i in range(mouse.shape[1]):
#          Train_coronal_024h.append(mouse[:,i,:])


# Test_coronal_001h = []
# Test_coronal_024h = []

# for timestamp in ["-001h", "024h"]:
#      mouse = "M08"
#      path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
#      ct = nib.load(path_ct).get_fdata()
#      for i in range(ct.shape[1]):
#         if timestamp == "-001h":
#             Test_coronal_001h.append(ct[:,i,:])
#         else:
#             Test_coronal_024h.append(ct[:,i,:])


