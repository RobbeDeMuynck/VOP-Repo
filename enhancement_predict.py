from enhancement_UNET import UNet
import torch
from enhancement_load_data import get_data
import matplotlib.pyplot as plt
import numpy as np
import json

# import nibabel as nib
# import pathlib


############################# LOADING THE MODEL  #############################
model_path = "MODELS\LYRS=3;FT=12;BS=4;LR=0.005;WD=0.pth"
# model_path = "MODELS\BS=8;LR=0.001;WD=0.09;FT=4.pth"
model_runlog = "runlogs\LYRS=3;FT=12;BS=4;LR=0.005;WD=0.json"

### plots presentation ###
good = 3, 16, 4, 0.001
bad = 4, 4, 12, 1e-5 #LYRS=4;FT=4;BS=12;LR=1e-05;WD=0

model_path = "MODELS\LYRS={};FT={};BS={};LR={};WD=0".format(*good) + ".pth"
model_runlog = "runlogs\LYRS={};FT={};BS={};LR={};WD=0".format(*good) + ".json"

# model_path = "MODELS\LYRS={};FT={};BS={};LR={};WD=0".format(*bad) + ".pth"
# model_runlog = "runlogs\LYRS={};FT={};BS={};LR={};WD=0".format(*bad) + ".json"

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
input, target, val_input, val_target = get_data(plane='sagittal', val_mouse=5)
ind = len(val_input)//2-10
slice_input, slice_target = normalize(val_input[ind]), normalize(val_target[ind])
slice_to_predict = torch.from_numpy(np.array(slice_input.copy())).unsqueeze(0).unsqueeze(0)

### APPLY MODEL ###
model.eval()
slice_prediction = torch.squeeze(model(slice_to_predict)[0]).detach().numpy()

# plot input vs prediction
fig, axs = plt.subplots(1, 3)
axs[0].imshow(slice_input, cmap='viridis')
axs[1].imshow(slice_target, cmap='viridis')
axs[2].imshow(slice_prediction, cmap='viridis')

axs[0].imshow(slice_input, cmap='bone')
axs[1].imshow(slice_target, cmap='bone')
axs[2].imshow(slice_prediction, cmap='bone')

axs[0].set_title('Before injection')
axs[0].axis('off')
axs[1].set_title('24 hours after injection with contrast enhancement')
axs[1].axis('off')
axs[2].set_title('Model prediction')
axs[2].axis('off')
plt.tight_layout()
# plt.savefig(f'IMAGES/PRED_SAG.png', dpi=200)
plt.show()


# ### Only for 3 layers ###
# after_conv1 = torch.squeeze(model(slice_to_predict)[1]).detach().numpy()
# after_pool1 = torch.squeeze(model(slice_to_predict)[2]).detach().numpy()
# after_conv2 = torch.squeeze(model(slice_to_predict)[3]).detach().numpy()
# after_pool2 = torch.squeeze(model(slice_to_predict)[4]).detach().numpy()
# after_conv3 = torch.squeeze(model(slice_to_predict)[5]).detach().numpy()
# after_pool3 = torch.squeeze(model(slice_to_predict)[6]).detach().numpy()
# after_bottle = torch.squeeze(model(slice_to_predict)[7]).detach().numpy()
# after_decoder1 = torch.squeeze(model(slice_to_predict)[8]).detach().numpy()
# after_decoder2 = torch.squeeze(model(slice_to_predict)[9]).detach().numpy()
# after_decoder3 = torch.squeeze(model(slice_to_predict)[10]).detach().numpy()
# ### PLOT RESULTS Only for 3 layers, plots first feature ###
# fig, axs = plt.subplots(2, 3)
# axs[0,0].imshow(slice_input, cmap='bone')
# axs[0,1].imshow(after_conv1[0], cmap='bone')
# axs[0,2].imshow(after_pool1[0], cmap='bone')
# axs[1,0].imshow(after_conv2[0], cmap='bone')
# axs[1,1].imshow(after_pool2[0], cmap='bone')
# axs[1,2].imshow(after_conv3[0], cmap='bone')

# axs[0,0].set_title('Input')
# axs[0,0].axis('off')
# axs[0,1].set_title('After Conv-block 1')
# axs[0,1].axis('off')
# axs[0,2].set_title('After Pooling 1')
# axs[0,2].axis('off')
# axs[1,0].set_title('After Conv-block 2')
# axs[1,0].axis('off')
# axs[1,1].set_title('After Pooling 2')
# axs[1,1].axis('off')
# axs[1,2].set_title('After Conv-block 3')
# axs[1,2].axis('off')
# plt.show()

# fig, axs = plt.subplots(2, 3)
# axs[0,0].imshow(after_pool3[0], cmap='bone')
# axs[0,1].imshow(after_bottle[0], cmap='bone')
# axs[0,2].imshow(after_decoder1[0], cmap='bone')
# axs[1,0].imshow(after_decoder2[0], cmap='bone')
# axs[1,1].imshow(after_decoder3[0], cmap='bone')
# axs[1,2].imshow(slice_prediction, cmap='bone')


# axs[0,0].set_title('After Pooling 3')
# axs[0,0].axis('off')
# axs[0,1].set_title('After Bottleneck')
# axs[0,1].axis('off')
# axs[0,2].set_title('After Decoder-block 1')
# axs[0,2].axis('off')
# axs[1,0].set_title('After Decoder-block 2')
# axs[1,0].axis('off')
# axs[1,1].set_title('After Decoder-block 3')
# axs[1,1].axis('off')
# axs[1,2].set_title('Model prediction')
# axs[1,2].axis('off')
# plt.show()

# ####

# # im_frame = Image.open("IMAGES/PF.png")
# # R, G, B = np.array(im_frame).T
# # IMG_before = np.array((R, G, B)).T
# # shape = IMG_before.shape
# # print(shape)
# # R = torch.from_numpy(np.array([R]).copy()).unsqueeze(0)
# # G = torch.from_numpy(np.array([G]).copy()).unsqueeze(0)
# # B = torch.from_numpy(np.array([B]).copy()).unsqueeze(0)

# # # plt.imshow(IMG_before)
# # # plt.show()

# # ############################# TESTING  #############################
# # model.eval()
# # R_ = np.array(model(R).detach().numpy()).reshape(3232,3232)
# # print('R ready')
# # G_ = np.array(model(G).detach().numpy()).reshape(3232,3232)
# # print('G ready')
# # B_ = np.array(model(B).detach().numpy()).reshape(3232,3232)
# # print('B ready')
# # IMG_after = np.array((R_, G_, B_)).T

# # fig = plt.subplots(figsize=(20,40))
# # plt.subplot(1,2,1)
# # plt.title('Before')
# # plt.imshow(IMG_before)
# # plt.subplot(1,2,2)
# # plt.title('After')
# # plt.imshow(IMG_after)
# # plt.show()

# ############################### sagittal in vertical #########################
# fig, axs = plt.subplots(3,1)
# axs[0].imshow(slice_input, cmap='viridis')
# axs[1].imshow(slice_target, cmap='viridis')
# axs[2].imshow(slice_prediction, cmap='viridis')

# axs[0].imshow(slice_input, cmap='bone')
# axs[1].imshow(slice_target, cmap='bone')
# axs[2].imshow(slice_prediction, cmap='bone')

# axs[0].set_title('Before injection')
# axs[0].axis('off')
# axs[1].set_title('24 hours after injection with contrast enhancement')
# axs[1].axis('off')
# axs[2].set_title('Model prediction')
# axs[2].axis('off')
# plt.tight_layout()
# # plt.savefig(f'IMAGES/PRED_SAG.png', dpi=200)
# plt.show()