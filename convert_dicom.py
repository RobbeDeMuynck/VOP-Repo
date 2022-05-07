###### in progress #######

import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids
from _UNET import UNet
import torch
from _load_data import get_data
import numpy as np
import json
import matplotlib.pyplot as plt

############################# LOADING THE MODEL  #############################
good = 3, 16, 4, 0.001

model_path = "MODELS\LYRS={};FT={};BS={};LR={};WD=0".format(*good) + ".pth"
model_runlog = "runlogs\LYRS={};FT={};BS={};LR={};WD=0".format(*good) + ".json"

with open(model_runlog, 'r') as RUN:
    run = json.load(RUN)
    layers, features = run["layers"], run["features"]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
torch.cuda.empty_cache()
model = UNet(layers, features).to(device)
model.load_state_dict(torch.load(model_path))



############################## LOAD IMAGES & NORMALIZE DATA ####################
def normalize(arr):
    return (arr-np.mean(arr))/np.std(arr)


input_transversal, target_transversal, val_input_transversal, val_target_transversal = get_data(plane='transversal', val_mouse=0)

scan = []
for input_slice, target_slice in zip(val_input_transversal, val_target_transversal):
    input_transversal, target_transversal = normalize(input_slice), normalize(target_slice)
    to_predict_transversal = torch.from_numpy(np.array(input_transversal.copy())).unsqueeze(0).unsqueeze(0)
    ### APPLY MODEL ###
    model.eval()
    prediction_transversal = torch.squeeze(model(to_predict_transversal)[0]).detach().numpy()
    ### place in 3d scan ###
    scan.append(prediction_transversal)
scan = np.asarray(scan)
print('Scan created')
# input_coronal, target_coronal, val_input_coronal,coronal, val_target_coronal = get_data(plane='coronal', val_mouse=0)
# input_coronal, target_coronal = normalize(val_input_coronal), normalize(val_target_coronal)
# to_predict_coronal = torch.from_numpy(np.array(input_coronal.copy())).unsqueeze(0).unsqueeze(0)

# input_sagittal, target_sagittal, val_input_sagittal,sagittal, val_target_sagittal = get_data(plane='sagittal', val_mouse=0)
# input_sagittal, target_sagittal = normalize(val_input_sagittal), normalize(val_target_sagittal)
# to_predict_sagittal = torch.from_numpy(np.array(input_sagittal.copy())).unsqueeze(0).unsqueeze(0)


# prediction_coronal = torch.squeeze(model(to_predict_coronal)[0]).detach().numpy()
# prediction_sagittal = torch.squeeze(model(to_predict_sagittal)[0]).detach().numpy()


################ create dicom image ###########################
def write_dicom(pixel_array,filename):

    meta = pydicom.Dataset()
    meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    ds.PatientName = "Mouse_03"
    ds.PatientID = "03"

    ds.Modality = "CT"
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid()

    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SamplesPerPixel = 1
    ds.HighBit = 15

    ds.ImagesInAcquisition = "1"

    ds.Rows = pixel_array.shape[0]
    ds.Columns = pixel_array.shape[1]
    ds.InstanceNumber = 1

    ds.ImagePositionPatient = r"0\0\1"
    ds.ImageOrientationPatient = r"1\0\0\0\-1\0"
    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.PixelSpacing = r"1\1"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 1

    pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
    ds.PixelData = pixel_array.tobytes()
    ds.save_as(r"Dicom scans/test5.dcm")
    return
# print(scan[100])
plt.imshow(scan[100], cmap='bone')
plt.show()
write_dicom(scan[100],'test')