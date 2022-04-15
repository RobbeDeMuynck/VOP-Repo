from UNET import *
import nibabel as nib
import pathlib
from PIL import Image
import numpy as np

############################# LOADING THE MODEL  #############################
model_path = "MODELS\BS=8;LR=0.001;WD=0.09;FT=4.pth"
device = torch.device('cpu')
model = UNet(4).to(device)
model.load_state_dict(torch.load(model_path))


### LOAD image ###
im_frame = Image.open("IMAGES/PF.png")
R, G, B = np.array(im_frame).T
IMG_before = np.array((R, G, B)).T
shape = IMG_before.shape
print(shape)
R = torch.from_numpy(np.array([R]).copy()).unsqueeze(0)
G = torch.from_numpy(np.array([G]).copy()).unsqueeze(0)
B = torch.from_numpy(np.array([B]).copy()).unsqueeze(0)

# plt.imshow(IMG_before)
# plt.show()

############################# TESTING  #############################
model.eval()
R_ = np.array(model(R).detach().numpy()).reshape(3232,3232)
print('R ready')
G_ = np.array(model(G).detach().numpy()).reshape(3232,3232)
print('G ready')
B_ = np.array(model(B).detach().numpy()).reshape(3232,3232)
print('B ready')
IMG_after = np.array((R_, G_, B_)).T

fig = plt.subplots(figsize=(20,40))
plt.subplot(1,2,1)
plt.title('Before')
plt.imshow(IMG_before)
plt.subplot(1,2,2)
plt.title('After')
plt.imshow(IMG_after)
plt.show()