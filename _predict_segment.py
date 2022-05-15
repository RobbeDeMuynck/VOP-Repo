from sklearn.preprocessing import OrdinalEncoder
# from torch import slice_scatter
from Segmentationnnn import *
import matplotlib.pyplot as plt
import numpy as np
import json
from torchsummary import summary
import seaborn as sns

#from mpl_toolkits.axes_grid1 import make_axes_locatable

# import nibabel as nib
# import pathlib

############################# LOADING THE MODEL  #############################
model_path = "MODELS/SEGG_layers4_lr0.001_wd0.01_ft16.pth"
# model_path = "MODELS\BS=8;LR=0.001;WD=0.09;FT=4.pth"
# model_runlog = "runlogs\LYRS=3;FT=12;BS=4;LR=0.005;WD=0.json"

# with open(model_runlog, 'r') as RUN:
#     run = json.load(RUN)
#     layers, features = run["layers"], run["features"]
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
#torch.cuda.empty_cache()
model = UNet(4, 16).to(device)
model.load_state_dict(torch.load(model_path))

H, W = 144, 112

### LOAD IMAGES & NORMALIZE DATA ###
def normalize(arr):
    return (arr-np.mean(arr))/np.std(arr)

input, target, val_input, val_target = get_data(val_mouse=5)
ind = 122
slice_input, slice_target = normalize(val_input[ind]), normalize(val_target[ind])
print('gelukt')

slice_to_predict = torch.from_numpy(np.array(slice_input.copy())).unsqueeze(0).unsqueeze(0)
print(f'modelinputsliceshape: {slice_to_predict.shape}')

### APPLY MODEL ###
model.eval()
x = model(slice_to_predict)[0]
slice_to_predict = torchvision.transforms.CenterCrop([H,W])(slice_to_predict)
slice_to_predict = torch.squeeze(slice_to_predict).detach().numpy()
#slice_to_predict = slice_to_predict[0:144,0:112]
print(f'imageshape:{slice_to_predict.shape}')

#print(x)

print(x.shape)

x = torch.squeeze(x).detach().numpy()
print(f'maskshape:{x.shape}')

slice_prediction =np.argmax(x,axis=0)




# plot input vs prediction
fig, axs = plt.subplots(1, 1)


# axs[0].imshow(slice_input, cmap='bone')
# axs[0].imshow(slice_target, cmap='viridis',alpha=.8)
axs.imshow(slice_to_predict,cmap='bone')

axs.imshow(slice_prediction, cmap='viridis',alpha=.6)
# divider = make_axes_locatable(axs[1])
# cax = divider.append_axes('right')

# fig.colorbar(seg, cax = cax)
# axs.add


#axs[0].set_title('Input')
axs.set_title('Prediction')
plt.tight_layout()
plt.savefig(f'IMAGES/PRED_SAG.png', dpi=200)
plt.show()
