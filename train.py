import torch
import numpy as np
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import Adam
from torch.autograd import Variable
from torchvision.transforms import transforms
import pathlib
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import seaborn as sns
import params

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

################################### DECLARING HYPERPARAMETERS  ##################################
num_epochs = params.num_epochs
batch_size = params.batch_size
learning_rate = params.learning_rate
weight_decay = params.weight_decay

################################### LOADING DATA TRANSVERSAAL  ##################################

Train_voor = []
Train_na = []

path = pathlib.Path('processed').parent
for timestamp in ["-001h", "024h"]:
     for mouse in ["M03", "M04", "M05", "M06", "M07"]:
         if timestamp == "-001h":
             path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
             Train_voor.append(nib.load(path_ct).get_fdata())
         else: 
             path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
             Train_na.append(nib.load(path_ct).get_fdata())           


Train_Data_001h = []
Train_Data_024h = []

for mouse in Train_voor:
    for i in range(mouse.shape[-1]):
        Train_Data_001h.append(mouse[:,:,i])


for mouse in Train_na:
    for i in range(mouse.shape[-1]):
         Train_Data_024h.append(mouse[:,:,i])


Test_Data_001h = []
Test_Data_024h = []

for timestamp in ["-001h", "024h"]:
     mouse = "M08"
     path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
     ct = nib.load(path_ct).get_fdata()
     for i in range(ct.shape[-1]):
        if timestamp == "-001h":
            Test_Data_001h.append(ct[:,:,i])
        else:
            Test_Data_024h.append(ct[:,:,i])

#Train_Data_001h (1210 slices)
#Train_Data_024h (1210 slices)
#Test_Data_001h  (242 slices)
#Test_Data_024h  (242 slices)


##################################### DEFINING BUILDING BLOCKS  ##################################

class conv_block(nn.Module): #dit is 1 blok van 2 convs gevolgd door een relu
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) #is dit noodzakelijk en waarom doet men dit en moet dit voor of na conv?

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
 
        self.relu = nn.ReLU() #evt leaky ReLu??

    def forward(self, inputs):
        inputs = inputs.float()
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)        
        return x

class res_block(nn.Module): #dit is 1 blok van 2 convs gevolgd door een relu
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) #is dit noodzakelijk en waarom doet men dit en moet dit voor of na conv?

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
 
        self.relu = nn.ReLU() #evt leaky ReLu??

        ###SKIP CONNECTION (Identity Mapping)
        self.s = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)

    def forward(self, inputs):
        inputs = inputs.float()
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        s = self.s(inputs)
        
        return x + s


class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = res_block(in_channels, out_channels)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        inputs = inputs.float()
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(in_channels, out_channels) #heb ik hier het juiste aantal channels???

    def forward(self, inputs, skip):
        inputs = inputs.float()
        skip = skip.float()
        x = self.up(inputs)
        _, _, H, W = x.shape
        skip = torchvision.transforms.CenterCrop([H,W])(skip)
        x = torch.cat([skip, x], axis=1) #ik heb ook al meer advanced versies van de resizing gezien, maakt dit veel uit?
        x = self.conv(x)
        return x

##################################### COMBINING BUILDING BLOCKS TO RESUNET##################################

class UNet(nn.Module):
    def __init__(self):
        super().__init__() #residuals nog implementeren.

        """ Encoder """
        self.e1 = encoder_block(1, 16)
        self.e2 = encoder_block(16, 32)
        self.e3 = encoder_block(32, 64)
        self.e4 = encoder_block(64, 128)

        """ Bottleneck """
        self.b = res_block(128, 256) # hoe beslis je eig hoeveel features je wilt per layer?

        """ Decoder """
        self.d1 = decoder_block(256,128)
        self.d2 = decoder_block(128,64)
        self.d3 = decoder_block(64,32)
        self.d4 = decoder_block(32,16)

        """ Last layer, i.e. de eigenlijke voorspelling """
        self.outputs = nn.Conv2d(16, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)
        return outputs


################################## INITIALIZING THE DATALOADER  ##################################

class MuizenDataset(Dataset):

    def __init__(self,data_voor,data_na,p=0.5):
        super().__init__()
        self.data_voor = (data_voor - np.mean(data_voor))/np.std(data_voor) #vanwege de kleine dataset laden we het gewoon helemaal in memory en normaliseren we in place
        self.data_na = (data_na - np.mean(data_na))-np.std(data_na)
        self.p = p

    def __len__(self):
        return len(self.data_voor)

    def __getitem__(self, index):

        input = self.data_voor[index]
        target = self.data_na[index]
        
        if torch.rand(1) < self.p:
            input = np.flipud(input)
            target = np.flipud(target)

        if torch.rand(1) < self.p:
            input = np.fliplr(input)
            target = np.fliplr(target)
        ###  DATA AUGMENTATION
        input = torch.from_numpy(input.copy())
        target = torch.from_numpy(target.copy())
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)
        
        return input.float(), target.float()
#onze laatste batch is incomplete dus deze laten we vallen


################################## TRAINING  ##################################

def MSE(input,output):
    return np.mean((input.detach().cpu().numpy()-output.detach().cpu().numpy())**2)


model = UNet().to(device)
optimizer = Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
loss_function = nn.MSELoss()
train_loader = DataLoader(MuizenDataset(Train_Data_001h,Train_Data_024h),batch_size=batch_size,shuffle=True,drop_last=True)
n_total_steps = len(train_loader)

model.train()


for epoch in range(num_epochs):  # we itereren meerdere malen over de data tot convergence?

    for i, (batch_voor, batch_na) in enumerate(tqdm(train_loader)): #wat is een handige manier om dit in te lezen?
        
        if torch.cuda.is_available(): #steek de batch in de GPU
            batch_voor=Variable(batch_voor.cuda())
            batch_na=Variable(batch_na.cuda())
        
        optimizer.zero_grad()
        predicted_batch = model(batch_voor)
        _, _, H, W = predicted_batch.shape
        batch_na = torchvision.transforms.CenterCrop([H,W])(batch_na)
        
        loss = loss_function(predicted_batch,batch_na) #vergelijk predicted na image met de echte na image
        loss.backward()
        optimizer.step()
        
        batch_voor = torchvision.transforms.CenterCrop([H,W])(batch_voor)

    print(f'Epoch: {epoch},         Loss: {loss.item()}')
            


##################################### SAVING THE MODEL  ##################################
model_path = "model.pth"
torch.save(model.state_dict(),model_path)
print('Model trained succesfully and saved as model.pth')