from Segmentationnnn import *
import matplotlib.pyplot as plt
import numpy as np
import json

# import nibabel as nib
# import pathlib


############################# LOADING THE MODEL  #############################
model_path = "MODELS\SEGG.pth"
# model_path = "MODELS\BS=8;LR=0.001;WD=0.09;FT=4.pth"
# model_runlog = "runlogs\LYRS=3;FT=12;BS=4;LR=0.005;WD=0.json"

# with open(model_runlog, 'r') as RUN:
#     run = json.load(RUN)
#     layers, features = run["layers"], run["features"]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
torch.cuda.empty_cache()
model = UNet(4, 16).to(device)
model.load_state_dict(torch.load(model_path))

### LOAD IMAGES & NORMALIZE DATA ###
def normalize(arr):
    return (arr-np.mean(arr))/np.std(arr)
input, target, val_input, val_target = get_data()
ind = len(val_input)//2
slice_input, slice_target = normalize(val_input[ind]), normalize(val_target[ind])
slice_to_predict = torch.from_numpy(np.array(slice_input.copy())).unsqueeze(0).unsqueeze(0)

### APPLY MODEL ###
model.eval()
slice_prediction = torch.squeeze(model(slice_to_predict)[0]).detach().numpy()[5]
print(slice_prediction.shape)
# plot input vs prediction
fig, axs = plt.subplots(1, 3)
axs[0].imshow(slice_input, cmap='viridis')
axs[1].imshow(slice_target, cmap='viridis')
axs[2].imshow(slice_prediction, cmap='viridis')

axs[0].set_title('Input')
axs[1].set_title('Target')
axs[2].set_title('Prediction')
plt.tight_layout()
plt.savefig(f'IMAGES/PRED_SAG.png', dpi=200)
plt.show()