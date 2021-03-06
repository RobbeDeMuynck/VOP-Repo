from sklearn.preprocessing import OrdinalEncoder
# from torch import slice_scatter
# from Segmentationnnn import *
from segmentation_load_data import MiceDataset, get_data
from segmentation_UNET import UNet
import matplotlib.pyplot as plt
import numpy as np
import json
from torchsummary import summary
import seaborn as sns
import torch
import torchvision
from report_tools.plots import segmentation_plot, segmentation_legend_elements

#from mpl_toolkits.axes_grid1 import make_axes_locatable

# import nibabel as nib
# import pathlib

### Crop function
def center_crop(input, H, W):
    _, y, x = input.shape
    start_y = y//2 - H//2
    start_x = x//2 - W//2
    return input[:, start_y:start_y+H, start_x:start_x+W]

def normalize(arr):
    return (arr-np.mean(arr))/np.std(arr)

############################# LOADING THE MODEL  #############################
# model_path = "MODELS/LargeSeg_layers4_lr0.001_wd0_ft12.pth"
model_path = "MODELS/SEGMENT_3lyrs_12fts.pth"
# model_path = "MODELS\SEGMENT_4lyrs_8fts_0.01LR.pth"
# model_path = "MODELS\BS=8;LR=0.001;WD=0.09;FT=4.pth"
# model_runlog = "runlogs\LYRS=3;FT=12;BS=4;LR=0.005;WD=0.json"

# with open(model_runlog, 'r') as RUN:
#     run = json.load(RUN)
#     layers, features = run["layers"], run["features"]
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
torch.cuda.empty_cache()
model = UNet(3, 12).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

### LOAD IMAGES & NORMALIZE DATA ###
train_input, train_target, val_input, val_target, test_input, test_target = get_data(plane='sagittal', val_mice=[15, 16, 17], test_mice=[18, 19, 20], standardize=True)

idx = 65
slice_input, slice_target = normalize(test_input[idx]), test_target[idx]

### APPLY MODEL ###
model.eval()
slice_to_predict = torch.from_numpy(np.array(slice_input.copy())).unsqueeze(0).unsqueeze(0)
organ_likelihood = model(slice_to_predict)[0]
organ_likelihood = torch.squeeze(organ_likelihood).detach().numpy()

slice_prediction_mask = np.argmax(organ_likelihood, axis=0)

### Make plot ###
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 3), constrained_layout=True)

segmentation_plot(ax1, test_input[idx], test_target[idx], legend=False)
segmentation_plot(ax2, test_input[idx], slice_prediction_mask, legend=False)

ax1.set_title('Actual organ masks')
ax2.set_title('Predicted organ masks')
# plt.legend(lines, labels, loc = 'lower center', bbox_to_anchor = (0,-0.1,1,1),
            # bbox_transform = plt.gcf().transFigure )
ax1.legend(handles=segmentation_legend_elements, loc='upper center', bbox_to_anchor=(0.8, -0.3), ncol=6)#, bbox_transform = plt.gcf().transFigure)

# fig, ax = plt.subplots(1, 1, figsize=(12, 4), constrained_layout=True, dpi=300)

# segmentation_plot(ax, test_input[idx], slice_prediction_mask, legend=False)

# ax1.set_title('Actual organ masks')
# ax2.set_title('Predicted organ masks')
# plt.legend(lines, labels, loc = 'lower center', bbox_to_anchor = (0,-0.1,1,1),
            # bbox_transform = plt.gcf().transFigure )
# ax.legend(handles=segmentation_legend_elements, loc='upper center', bbox_to_anchor=(0.8, -0.3), ncol=6)#, bbox_transform = plt.gcf().transFigure)
# ax.legend(handles=segmentation_legend_elements, loc='center', ncol=6)#, bbox_transform = plt.gcf().transFigure)


plt.tight_layout()
plt.savefig("IMAGES/segmentation_report.png")
plt.show()



# #slice_to_predict = slice_to_predict[0:144,0:112]
# print(f'imageshape:{slice_to_predict.shape}')

# # plot input vs prediction
# fig, ax = plt.subplots(1, 1)


# # axs[0].imshow(slice_input, cmap='bone')
# # axs[0].imshow(slice_target, cmap='viridis',alpha=.8)
# axs.imshow(slice_to_predict,cmap='bone')

# axs.imshow(slice_prediction, cmap='viridis',alpha=.6)
# # divider = make_axes_locatable(axs[1])
# # cax = divider.append_axes('right')

# # fig.colorbar(seg, cax = cax)
# # axs.add


# #axs[0].set_title('Input')
# axs.set_title('Prediction')
# # plt.tight_layout()
# # plt.savefig(f'IMAGES/PRED_SAG.png', dpi=200)
# plt.show()
