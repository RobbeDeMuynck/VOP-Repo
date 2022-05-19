from enhancement_UNET import UNet
import torch
import torch.nn as nn
from enhancement_load_data import get_data
import matplotlib.pyplot as plt
import numpy as np
import json

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

from torchsummary import summary

############################# LOADING THE MODEL  #############################
model_path = "MODELS\LYRS=4;FT=12;BS=4;LR=0.005;WD=0.pth"
# model_path = "MODELS\BS=8;LR=0.001;WD=0.09;FT=4.pth"
model_runlog = "runlogs\LYRS=4;FT=12;BS=4;LR=0.005;WD=0.json"

### plots presentation ###
good = 3, 16, 4, 0.001
bad = 4, 4, 12, 1e-5 #LYRS=4;FT=4;BS=12;LR=1e-05;WD=0

model_path = "MODELS\LYRS={};FT={};BS={};LR={};WD=0".format(*good) + ".pth"
model_runlog = "runlogs\LYRS={};FT={};BS={};LR={};WD=0".format(*good) + ".json"

with open(model_runlog, 'r') as RUN:
    run = json.load(RUN)
    layers, features = run["layers"], run["features"]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
torch.cuda.empty_cache()
model = UNet(layers, features).to(device)
model.load_state_dict(torch.load(model_path))

### LOAD IMAGES & NORMALIZE DATA ###
def normalize(arr):
    return (arr-np.mean(arr))/np.std(arr)
input, target, val_input, val_target = get_data(plane='transversal', val_mouse=5)
ind = 117
slice_input, slice_target = normalize(val_input[ind]), normalize(val_target[ind])
slice_to_predict = torch.from_numpy(np.array(slice_input.copy())).unsqueeze(0).unsqueeze(0)
print(slice_to_predict.shape)

# To assist you in designing the feature extractor you may want to print out
# the available nodes
# print(model)

summary(model, (1, 154, 121))

# layers = []
# model_children = list(model.children())
# for child in model_children:
#     print(child)
#     if type(child) == nn.Conv2d:
#         layers.append(child)
#     # elif type(child) == nn.Sequential:
#     #     for layer in child.children():
#     #         if type(layer) == nn.Conv2d:
#     #             layers.append(layer)
# print(model_children)
# print(layers)
# return_nodes = ['d1.conv2']
# feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
# train_nodes, eval_nodes = get_graph_node_names(UNet(layers, features))
# return_nodes = {'e1': '1'}

# create_feature_extractor(model, return_nodes=return_nodes)


