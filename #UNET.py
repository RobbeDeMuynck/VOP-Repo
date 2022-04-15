import torch
import torch.nn as nn 
import torchvision
import numpy as np
from torch.utils.data import Dataset

# import torch
# import numpy as np
# import torch.nn as nn
# import torchvision
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from torch.optim import Adam
# from torch.autograd import Variable
# from torchvision.transforms import transforms
# # import pathlib
# # import nibabel as nib
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
# import time
# import seaborn as sns
# import params

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# torch.cuda.empty_cache()


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

    def __init__(self, ft=10, layers=4):
        super().__init__()
        self.layers = layers

        ### Encoder ###
        self.encoders = []
        self.encoders.append(encoder_block(1, ft))
        for i in range(1, layers):
            self.encoders.append(encoder_block(2**(i-1)*ft, 2**(i)*ft))

        ### Bottleneck ###
        self.bottleneck = res_block(2**(layers-1)*ft, 2**(layers)*ft)
        
        ### Decoder ###
        self.decoders = []
        for i in range(layers, 0, -1):
            self.decoders.append(decoder_block(2**(i)*ft, 2**(i-1)*ft))
            
        ### Last layer: mapping to prediction ###
        self.outputs = nn.Conv2d(ft, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        ### Encoder ###
        # Store skip connections in dictionary for later use in decoder
        skip_con = {}
        down = inputs
        for i, encoder in enumerate(self.encoders):
            skip_con[f'{i}'], down = encoder(down)

        ### Bottleneck ###
        bottleneck = self.bottleneck(down)

        ### Decoder ###
        up = bottleneck
        for i, decoder in enumerate(self.decoders):
            up = decoder(up, skip_con[f'{self.layers-1-i}'])

        ### Last layer: mapping to prediction ###
        outputs = self.outputs(up)
        return outputs

################################## INITIALIZING THE DATALOADER  ##################################

class MiceDataset(Dataset):
    def __init__(self, data_voor, data_na, p=0.5):
        super().__init__()
        self.data_voor = (data_voor - np.mean(data_voor))/np.std(data_voor) #vanwege de kleine dataset laden we het gewoon helemaal in memory en normaliseren we in place
        self.data_na = (data_na - np.mean(data_na))-np.std(data_na)
        self.p = p

    def __len__(self):
        return len(self.data_voor)

    def __getitem__(self, index):

        input = self.data_voor[index]
        target = self.data_na[index]
        
        ###  DATA AUGMENTATION
        if torch.rand(1) < self.p:
            input = np.flipud(input)
            target = np.flipud(target)

        if torch.rand(1) < self.p:
            input = np.fliplr(input)
            target = np.fliplr(target)
        
        input = torch.from_numpy(input.copy()).unsqueeze(0).float()
        target = torch.from_numpy(target.copy()).unsqueeze(0).float()
        return input, target
