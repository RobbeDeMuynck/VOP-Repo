import torch
import torch.nn as nn 
import torchvision

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
    -- layers: (int) number of encoding operations in the network
    -- ft: (int) number of starting features: number of channels of the first layer"""

    def __init__(self, layers=3, ft=12):
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
            self.outputs = nn.Conv2d(ft, 1, kernel_size=1, padding=0)
        else: 
            raise Exception("Cannot construct networ: 'layers' parameter can only be 3 or 4")

    def forward(self, inputs):
        if self.layers == 3:
            ### Encoder ###
            # Store skip connections in dictionary for later use in decoder
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