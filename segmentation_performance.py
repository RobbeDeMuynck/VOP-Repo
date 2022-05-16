# from Segmentationnnn import *
import torch
from UNET_segmentation import UNet
from load_data_segmentation import MiceDataset, get_data
import numpy as np
import json
from torchsummary import summary
from report_tools.confusion_matrix import CM_plot
from tqdm import tqdm

### Crop function
def center_crop(input, H, W):
    _, y, x = input.shape
    start_y = y//2 - H//2
    start_x = x//2 - W//2
    return input[:, start_y:start_y+H, start_x:start_x+W]

def normalize(arr):
    return (arr-np.mean(arr))/np.std(arr)

### Load the model
model_path = "MODELS/SEGG_layers4_lr0.001_wd0.01_ft16.pth"
model_path = "MODELS/SEGMENT_3lyrs_12fts.pth"

# model_path = "MODELS\BS=8;LR=0.001;WD=0.09;FT=4.pth"
# model_runlog = "runlogs\LYRS=3;FT=12;BS=4;LR=0.005;WD=0.json"

# with open(model_runlog, 'r') as RUN:
#     run = json.load(RUN)
#     layers, features = run["layers"], run["features"]
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

device = torch.device('cpu')
torch.cuda.empty_cache()
model = UNet(3, 12).to(device)
model.load_state_dict(torch.load(model_path))

### Read-in data
train_input, train_target, val_input, val_target, test_input, test_target = get_data(plane='sagittal', val_mice=[15, 16, 17], test_mice=[18, 19, 20], standardize=True)
test_input_normalize, target_masks = normalize(test_input), test_target
# test_input_normalize, target_masks = normalize(train_input), train_target

### APPLY MODEL ###
model.eval()
# Select bounds
bounds = False
if bounds is False:
    start, stop = 0, -1
else:
    start, stop = bounds

# Make prediction for each slice, depending on bounds
prediction_masks = []
for slice_to_predict in tqdm(test_input_normalize[start:stop]):
    slice_to_predict = torch.from_numpy(np.array(slice_to_predict.copy())).unsqueeze(0).unsqueeze(0)
    organ_likelihood = model(slice_to_predict)[0]
    organ_likelihood = torch.squeeze(organ_likelihood).detach().numpy()
    
    slice_prediction_mask = np.argmax(organ_likelihood, axis=0)

    prediction_masks.append(slice_prediction_mask)

prediction_masks = np.array(prediction_masks)
#  H, W = 144, 112
W, H, L = prediction_masks.shape
target_masks_crop = center_crop(target_masks[start:stop], H, L)

### Plot confusion matrix
CM_plot(target_masks_crop, prediction_masks)