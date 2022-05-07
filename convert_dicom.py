###### in progress #######

import dicom, dicom.UID
from dicom.dataset import Dataset, FileDataset
from _UNET import UNet
import torch
from _load_data import get_data
import numpy as np
import json

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

input_transversal, target_transversal, val_input_transversal,transversal, val_target_transversal = get_data(plane='transversal', val_mouse=0)
input_transversal, target_transversal = normalize(val_input_transversal), normalize(val_target_transversal)
to_predict_transversal = torch.from_numpy(np.array(input_transversal.copy())).unsqueeze(0).unsqueeze(0)

input_coronal, target_coronal, val_input_coronal,coronal, val_target_coronal = get_data(plane='coronal', val_mouse=0)
input_coronal, target_coronal = normalize(val_input_coronal), normalize(val_target_coronal)
to_predict_coronal = torch.from_numpy(np.array(input_coronal.copy())).unsqueeze(0).unsqueeze(0)

input_sagittal, target_sagittal, val_input_sagittal,sagittal, val_target_sagittal = get_data(plane='sagittal', val_mouse=0)
input_sagittal, target_sagittal = normalize(val_input_sagittal), normalize(val_target_sagittal)
to_predict_sagittal = torch.from_numpy(np.array(input_sagittal.copy())).unsqueeze(0).unsqueeze(0)

### APPLY MODEL ###
model.eval()
prediction_transversal = torch.squeeze(model(to_predict_transversal)[0]).detach().numpy()
prediction_coronal = torch.squeeze(model(to_predict_coronal)[0]).detach().numpy()
prediction_sagittal = torch.squeeze(model(to_predict_sagittal)[0]).detach().numpy()


def write_dicom(pixel_array,filename):
    
    file_meta = Dataset()
    ds = FileDataset(filename, {},file_meta = file_meta,preamble="\0"*128)
    ds.PixelData = pixel_array.tostring()
    ds.save_as(filename)
    return
