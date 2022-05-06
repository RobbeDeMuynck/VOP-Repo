import numpy as np
import torch
from torch.utils.data import Dataset
import pathlib
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 
import torchvision
from _UNET import UNet
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import time
import json

from _load_data import MiceDataset, get_data
from torch.utils.data import DataLoader


path = pathlib.Path(__file__).parent

# layers = 4
# features = 64

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

def get_data(val_mouse = 5):
    train_input, train_target = [],[]
    val_input, val_target = [],[]
    mice = ["M03", "M04", "M05", "M06", "M07","M08"]
    train_names = [mouse for i, mouse in enumerate(mice) if i!= val_mouse]
    val_names = [mouse for i, mouse in enumerate(mice) if i == val_mouse]
    for mouse in train_names:
        for timestamp in ["024h"]:
            path_ct = path / f"original/{mouse}_{timestamp}/CT280.img"
            path_organ = path / f"original/{mouse}_{timestamp}/Organ_280.img"
            path_class = path / f"original/{mouse}_{timestamp}/Organ_280.cls"
            if not path_organ.is_file():
                path_organ = path / f"original/{mouse}_{timestamp}/Organ1_280.img"
            if not path_class.is_file():
                path_class = path / f"original/{mouse}_{timestamp}/Organ1_280.cls"
            
            ct = nib.load(path_ct).get_fdata()
            organ = nib.load(path_organ).get_fdata()
            for i in range(ct.shape[-1]):
                train_input.append(ct[:,:,i])
                train_target.append(organ[:,:,i])
            #print(f'CT size:{ct.shape}')

    for mouse in val_names:
        for timestamp in ["024h"]:
            path_ct = path / f"original/{mouse}_{timestamp}/CT280.img"
            path_organ = path / f"original/{mouse}_{timestamp}/Organ_280.img"
            path_class = path / f"original/{mouse}_{timestamp}/Organ_280.cls"
            if not path_organ.is_file():
                path_organ = path / f"original/{mouse}_{timestamp}/Organ1_280.img"
            if not path_class.is_file():
                path_class = path / f"original/{mouse}_{timestamp}/Organ1_280.cls"

            ct = nib.load(path_ct).get_fdata()

            organ = nib.load(path_organ).get_fdata()
            for i in range(ct.shape[-1]):
                val_input.append(ct[:,:,i])
                val_target.append(organ[:,:,i])

    x1,y1,x2,y2 = np.array(train_input), np.array(train_target,dtype=int), np.array(val_input), np.array(val_target,dtype=int)
    # print(x1.shape,y1.shape,x2.shape,y2.shape)
    # print("Data Initialized")
    # print("/n")
    # print("/n")
    # print("/n")

    # print("/n")
    return x1,y1,x2,y2

        # with open(path_class) as f:
        #     # EXAMPLE for "M03_024h/Organ_280.cls":
        #     # ClassColors=0 0 0 255|116 161 166 255|0 85 0 255|201 238 255 255|255 170 255 255|0 0 255 255|176 230 241 255|0 130 182 255|71 205 108 255|0 255 0 255|0 255 255 255|56 65 170 255|175 235 186 255
        #     # ClassIndices=0|1|2|3|4|5|6|7|8|9|10|11|12
        #     # ClassNames=unclassified|Trachea|Spleen|Bone|Lung|Heart|Stomach|Bladder|Muscle|Tumor|Kidneys|Liver|Intestine

# def to_categorical(y, num_classes):
#     """ 1-hot encodes a tensor """
#     return np.eye(num_classes, dtype='uint8')[y]

class MiceDataset(Dataset):
    def __init__(self, data_in, target, p=0.5):
        super().__init__()
        ### Normalize data ###
        self.data_in = (data_in - np.mean(data_in))/np.std(data_in)
        self.target = target
        self.p = p
    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, index):
        input = self.data_in[index]
        target = self.target[index] #maak er een one-hotencoding van

        if torch.rand(1) < self.p:
            input = np.flipud(input)
            target = np.flipud(target)

        if torch.rand(1) < self.p:
            input = np.fliplr(input)
            target = np.fliplr(target)

        #print(f'input dataloader:{input.shape},{target.shape}')
        input = torch.from_numpy(input.copy()).unsqueeze(0).float()
        target = torch.from_numpy(target.copy()).float()
        #torch.reshape(target,(target.shape[-1],target.shape[0],target.shape[1]))
        
        return input, target



##################################### DEFINING BUILDING BLOCKS  ##################################
class conv_block(nn.Module):
    """2 times:  3x3 convolution, followed by batch normalization and a ReLu."""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) 
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = inputs.float()
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x ### Hier geen identity mapping?

class res_block(nn.Module):
    """2 times:  3x3 convolution, followed by batch normalization and a ReLu.
    After these operations, the input is added (skip connection)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        ### SKIP CONNECTION (Identity Mapping)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        inputs = inputs.float()
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        s = self.skip(inputs)
        return x + s

class encoder_block(nn.Module):
    """Downscaling: double convolution (with identity mapping), followed by 2x2 maxpool."""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = res_block(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        inputs = inputs.float()
        x = self.double_conv(inputs)
        p = self.maxpool(x)
        return x, p

class decoder_block(nn.Module):
    """Upscaling: transposed convolution, skip connection of encoding side,
    followed by double convolution (without identity mapping)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.double_conv = conv_block(in_channels, out_channels)

    def forward(self, inputs, skip):
        inputs = inputs.float()
        skip = skip.float()
        x = self.up(inputs)
        _, _, H, W = x.shape
        skip = torchvision.transforms.CenterCrop([H,W])(skip)
        x = torch.cat([skip, x], axis=1)
        x = self.double_conv(x)
        return x

##################################### COMBINING BUILDING BLOCKS TO RESUNET##################################
class UNet(nn.Module):
    """A UNet neural network is constructed by combining its subblocks.
    The network structure is initialized by declaring the following parameters:
    -- ft: (int) number of starting features: number of channels of the first layer
    -- layers: (int) number of encoding operations in the network"""

    def __init__(self, layers=3, ft=64):
        super().__init__()
        self.layers = layers

        if layers == 3:
            ### Encoder ###
            self.e1 = encoder_block(1, ft)
            self.e2 = encoder_block(ft, 2*ft)
            self.e3 = encoder_block(2*ft, 4*ft)

            ### Bottleneck ###
            self.b = res_block(4*ft, 8*ft)
            
            ### Decoder ###
            self.d1 = decoder_block(8*ft,4*ft)
            self.d2 = decoder_block(4*ft,2*ft)
            self.d3 = decoder_block(2*ft,ft)

            ### Last layer: mapping to prediction ###
            self.outputs = nn.Conv2d(ft, 1, kernel_size=1, padding=0)
        elif layers == 4:
            ### Encoder ###
            self.e1 = encoder_block(1, ft)
            self.e2 = encoder_block(ft, 2*ft)
            self.e3 = encoder_block(2*ft, 4*ft)
            self.e4 = encoder_block(4*ft, 8*ft)

            ### Bottleneck ###
            self.b = res_block(8*ft, 16*ft)
            
            ### Decoder ###
            self.d1 = decoder_block(16*ft,8*ft)
            self.d2 = decoder_block(8*ft,4*ft)
            self.d3 = decoder_block(4*ft,2*ft)
            self.d4 = decoder_block(2*ft,ft)

            ### Last layer: mapping to prediction ###
            self.outputs = nn.Conv2d(ft, 13, kernel_size=1, padding=0)
        else: 
            raise Exception("Cannot construct networ: 'layers' parameter can only be 3 or 4")

    def forward(self, inputs):
        if self.layers == 3:
            ### Encoder ###
            # Store skip connections in dictionary for later use in decoder
            #print(f'inputsizeUNET: {inputs.shape}')
            s1, p1 = self.e1(inputs)
            s2, p2 = self.e2(p1)
            s3, p3 = self.e3(p2)

            ### Bottleneck ###
            b = self.b(p3)

            ### Decoder ###
            d1 = self.d1(b, s3)
            d2 = self.d2(d1, s2)
            d3 = self.d3(d2, s1)

            ### Last layer: mapping to prediction ###
            outputs = self.outputs(d3)
            return outputs,s1,p1,s2,p2,s3,p3,b,d1,d2,d3

        elif self.layers == 4:
            ### Encoder ###
            # Store skip connections in dictionary for later use in decoder
            s1, p1 = self.e1(inputs)
            s2, p2 = self.e2(p1)
            s3, p3 = self.e3(p2)
            s4, p4 = self.e4(p3)

            ### Bottleneck ###
            b = self.b(p4)

            ### Decoder ###
            d1 = self.d1(b, s4)
            d2 = self.d2(d1, s3)
            d3 = self.d3(d2, s2)
            d4 = self.d4(d3, s1)

            ### Last layer: mapping to prediction ###
            outputs = self.outputs(d4)
            return outputs,s1,p1,s2,p2,s3,p3,s4,p4,b,d1,d2,d3,d4
        
# train_in,train_tar,_,_ = get_data()
# data = MiceDataset(train_in,train_tar)
# data.__getitem__(500)
# model = UNet(layers=layers, ft=features)
# pred = model(data.__getitem__(500)[0])
