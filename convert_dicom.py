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
bad = 4, 4, 12, 1e-5 #LYRS=4;FT=4;BS=12;LR=1e-05;WD=0

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



### LOAD IMAGES & NORMALIZE DATA ###
def normalize(arr):
    return (arr-np.mean(arr))/np.std(arr)

input, target, val_input, val_target = get_data(plane='transversal', val_mouse=0)
input, target = normalize(val_input), normalize(val_target)
to_predict = torch.from_numpy(np.array(input.copy())).unsqueeze(0).unsqueeze(0)

### APPLY MODEL ###
model.eval()
prediction = torch.squeeze(model(to_predict)[0]).detach().numpy()

def write_dicom(pixel_array,filename):
    
    file_meta = Dataset()
    ds = FileDataset(filename, {},file_meta = file_meta,preamble="\0"*128)
    ds.PixelData = pixel_array.tostring()
    ds.save_as(filename)
    return

