import pathlib
from tkinter.messagebox import NO
import nibabel as nib
import torch
import numpy as np
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import Adam
from torch.autograd import Variable
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import params
from tracemalloc import start
from turtle import begin_fill
import params
import MiceData
from UNETleaky import *
import json
import time
from datetime import datetime

##################################### LOADING DATA TRANSVERSAL  #######################
##################################### LOADING DATA TRANSVERSAL  #######################
##################################### LOADING DATA TRANSVERSAL  #######################

########################### BUILDING UNET #############################################
########################### BUILDING UNET #############################################
########################### BUILDING UNET #############################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
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

        ### SKIP CONNECTION (Identity Mapping)
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

#COMBINING BUILDING BLOCKS TO RESUNET 

class UNet(nn.Module):
    def __init__(self, f1=10):
        super().__init__()

        f1 = params.features

        """ Encoder """
        self.e1 = encoder_block(1, f1)
        self.e2 = encoder_block(f1, 2*f1)
        self.e3 = encoder_block(2*f1, 4*f1)
        self.e4 = encoder_block(4*f1, 8*f1)

        """ Bottleneck """
        self.b = res_block(8*f1, 16*f1) # hoe beslis je eig hoeveel features je wilt per layer?

        """ Decoder """
        self.d1 = decoder_block(16*f1,8*f1)
        self.d2 = decoder_block(8*f1,4*f1)
        self.d3 = decoder_block(4*f1,2*f1)
        self.d4 = decoder_block(2*f1,f1)

        """ Last layer, i.e. de eigenlijke voorspelling """
        self.outputs = nn.Conv2d(f1, 1, kernel_size=1, padding=0)

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

# INITIALIZING THE DATALOADER
class MuizenDataset(Dataset):

    def __init__(self, data_voor, data_na, p=0.5):
        super().__init__()
        self.data_voor = (np.asarray(data_voor) - np.mean(np.asarray(data_voor)))/np.std(np.asarray(data_voor))
        self.data_na = (np.asarray(data_na) - np.mean(np.asarray(data_na)))/np.std(np.asarray(data_na))
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
        
        input = torch.from_numpy(input.copy())
        target = torch.from_numpy(target.copy())
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)
        
        return input.float(), target.float()
# onze laatste batch is incomplete dus deze laten we vallen

loss_function = nn.MSELoss()

############################# TRAINING & TESTING  #############################
############################# TRAINING & TESTING  #############################
############################# TRAINING & TESTING  #############################
class LieveMuizen():
    def __init__(self,view = 'transverse',test_mouse = 0,num_epochs=120, batch_size=4, learning_rate=0.01, weight_decay=0.001, features=16,patience=5, save=True):
        #test mouse kan index 0-5 hebben en is de muis waarop je wilt testen, trainen gebeurt op de andere muizen.
        self.test_mouse = test_mouse
        self.save=save
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.features = features
        self.save = save
        self.train_input = None
        self.train_target = None
        self.test_input = None
        self.test_target = None
        self.model_path = None
        self.model_name = None
        self.view = view

    def prep_data(self): 
        Train_voor = []
        Train_na = []
        mouses = ["M03", "M04", "M05", "M06", "M07","M08"]
        train_mouses = [mouses[i] for i in range(len(mouses)) if i!= self.test_mouse]
        test_muisje = mouses[self.test_mouse]
        path = pathlib.Path('processed').parent
        for timestamp in ["-001h", "024h"]:
            for mouse in train_mouses:
                if timestamp == "-001h":
                    path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
                    Train_voor.append(nib.load(path_ct).get_fdata())
                else: 
                    path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
                    Train_na.append(nib.load(path_ct).get_fdata())           
        if self.view == 'transverse':

            Train_transversal_001h = []
            Train_transversal_024h = []

            for mouse in Train_voor:
                for i in range(mouse.shape[-1]):
                    Train_transversal_001h.append(mouse[:,:,i])

            for mouse in Train_na:
                for i in range(mouse.shape[-1]):
                    Train_transversal_024h.append(mouse[:,:,i])

            Test_transversal_001h = []
            Test_transversal_024h = []

            for timestamp in ["-001h", "024h"]:
                mouse = test_muisje
                path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
                ct = nib.load(path_ct).get_fdata()
                for i in range(ct.shape[-1]):
                    if timestamp == "-001h":
                        Test_transversal_001h.append(ct[:,:,i])
                    else:
                        Test_transversal_024h.append(ct[:,:,i])
            print('Data successfully initialized')
            self.train_input,self.train_target,self.test_input,self.test_target = Train_transversal_001h,Train_transversal_024h,Test_transversal_001h,Test_transversal_024h
            return None
    # Train_transversal_001h (1210 slices)
    # Train_transversal_024h (1210 slices)
    # Test_transversal_001h  (242 slices)
    # Test_transversal_024h  (242 slices)

    ##################################### LOADING DATA SAGITTAL ##################################
        elif self.view=='sagittal':
            Train_sagittal_001h = []
            Train_sagittal_024h = []

            for mouse in Train_voor:
                for i in range(mouse.shape[1]):
                    Train_sagittal_001h.append(mouse[:,i,:])

            for mouse in Train_na:
                for i in range(mouse.shape[1]):
                    Train_sagittal_024h.append(mouse[:,i,:])

            Test_sagittal_001h = []
            Test_sagittal_024h = []

            for timestamp in ["-001h", "024h"]:
                mouse = test_muisje
                path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
                ct = nib.load(path_ct).get_fdata()
                for i in range(ct.shape[-1]):
                    if timestamp == "-001h":
                        Test_sagittal_001h.append(ct[:,i,:])
                    else:
                        Test_sagittal_024h.append(ct[:,i,:])
            self.train_input,self.train_target,self.test_input,self.test_target = Train_sagittal_001h,Train_sagittal_024h,Test_sagittal_001h,Test_sagittal_024h
            print('Data successfully initialized')
            return None

    #Train_sagittal_001h (500 slices)
    #Train_sagittal_024h (500 slices)
    #Test_sagittal_001h  (100 slices)
    #Test_sagittal_024h  (100 slices)

        elif self.view=='coronal':
            Train_coronal_001h = []
            Train_coronal_024h = []

            for mouse in Train_voor:
                for i in range(mouse.shape[0]):
                    Train_coronal_001h.append(mouse[i,:,:])

            for mouse in Train_na:
                for i in range(mouse.shape[0]):
                    Train_coronal_024h.append(mouse[i,:,:])

            Test_coronal_001h = []
            Test_coronal_024h = []

            for timestamp in ["-001h", "024h"]:
                mouse = test_muisje
                path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
                ct = nib.load(path_ct).get_fdata()
                for i in range(ct.shape[-1]):
                    if timestamp == "-001h":
                        Test_coronal_001h.append(ct[i,:,:])
                    else:
                        Test_coronal_024h.append(ct[i,:,:])
            print('Data successfully initialized')
            self.train_input,self.train_target,self.test_input,self.test_target =  Train_coronal_001h,Train_coronal_024h,Test_coronal_001h,Test_coronal_024h
            return None
        print('Data loading failed')
        return None


    def train(self):
        model = UNet(self.features).to(device)
        optimizer = Adam(
                        model.parameters(),
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay
                        )
        train_loader = DataLoader(
                        MuizenDataset(self.train_input, self.train_target),
                        batch_size=self.batch_size,
                        shuffle=True,
                        drop_last=True
                        )

        # test_loader = DataLoader(
        #                 MuizenDataset(?, ?),
        #                 batch_size=batch_size,
        #                 shuffle=True,
        #                 )
        val_loader = DataLoader(
                        MuizenDataset(self.test_input, self.test_target),#### Hier heb je toch dataleaking????
                        batch_size=self.batch_size,
                        shuffle=True,
                        drop_last=True
                        )
        # n_train = len(train_loader)

        print('Starting with training...')
        loss_stats = {
            'train': [],
            'val': []
            }

        best_loss = np.inf
        starttime = time.time()
        for epoch in range(1, self.num_epochs+1):  # we itereren meerdere malen over de data tot convergence?
            model.train()
            train_epoch_loss = 0
            print(f"Epoch: {epoch}/{self.num_epochs}")
            for i, (input_batch, target_batch) in enumerate(tqdm(train_loader)): #wat is een handige manier om dit in te lezen?
                
                if torch.cuda.is_available(): #steek de batch in de GPU
                    input_batch = Variable(input_batch.cuda())
                    target_batch = Variable(target_batch.cuda())
                
                optimizer.zero_grad()
                prediction_batch = model(input_batch)
                _, _, H, W = prediction_batch.shape
                target_batch = torchvision.transforms.CenterCrop([H,W])(target_batch)
                
                loss = loss_function(prediction_batch, target_batch) #vergelijk predicted na image met de echte na image
                loss.backward()
                optimizer.step()

                train_epoch_loss += loss.item()
                
                input_batch = torchvision.transforms.CenterCrop([H,W])(input_batch)
            
            # Prevent overfitting
            with torch.no_grad():
                val_epoch_loss = 0
                    
                model.eval()
                for val_input_batch, val_target_batch in val_loader:
                    val_input_batch = val_input_batch.to(device)
                    val_target_batch = val_target_batch.to(device)

                    val_pred = model(val_input_batch)
                    val_target_batch = torchvision.transforms.CenterCrop([H,W])(val_target_batch)

                    val_loss = loss_function(val_pred, val_target_batch)
                    val_epoch_loss += val_loss.item()
            
            loss_stats["train"].append(train_epoch_loss/len(train_loader))
            loss_stats["val"].append(val_epoch_loss/len(val_loader))
            
            if loss_stats["val"][-1] < best_loss:
                endtime = time.time()
                best_loss, epoch_no = loss_stats["val"][-1], epoch
                if self.save == True:
                    self.model_name = datetime.now().strftime("%Y_%m_%d-%I-%M-%S_%p")+f'__BS_{self.batch_size}__features_{self.features}_testmuis__{self.test_mouse}__{self.view}'
                    self.model_path = 'MODELS/'+self.model_name
                    torch.save(model.state_dict(), self.model_path+'.pth')
                    print(f"    --> Model saved at epoch no. {epoch_no}")
            
            print(f"""
        Training loss: {loss_stats["train"][-1]}
        Last validation losses: {loss_stats['val'][-self.patience:]}
        Best loss: {best_loss}""")

            if np.all(np.array(loss_stats['val'][-self.patience:]) > best_loss):
                print(f"""
        Training terminated after epoch no. {epoch}
        Model saved as {self.model_name}: version at epoch no. {epoch_no}""")
                break
            elif epoch == self.num_epochs:
                epoch_no = epoch

        # print(f"""Training_losses = {loss_stats["train"]}
        # Validation_losses = {loss_stats["val"]}""")

        # Dictionary with model details
        run = {
            'train_time' : endtime-starttime,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'patience': self.patience,
            'features': self.features,
            'train_loss': loss_stats["train"],
            'val_loss': loss_stats["val"], 
            'num_epoch_convergence': epoch_no
        }

        with open(f'runlogs/{self.model_name}.json', 'w+') as file:
                    json.dump(run, file, indent=4)
        print(f'Model successfully trained and saved to {self.model_path}.pth')
        return run

    def test(self,plot=True):
        input = self.test_input
        target = self.test_target
        model = UNet(self.features).to(device)
        model.load_state_dict(torch.load(self.model_path+'.pth'))
        test_loader = DataLoader(
                        MuizenDataset(input, target),
                        batch_size=self.batch_size,
                        shuffle=True,
                        drop_last=True
                        )
        model.eval()
        print('Starting with testing...')

        losses = []
        for i, (input_batch,target_batch) in enumerate(tqdm(test_loader)):
            #input_batch = input_batch.view(batch_size,1,121,242)
            #target_batch = target_batch.view(batch_size,1,121,242)
            
            if torch.cuda.is_available():
                input_batch=Variable(input_batch.cuda())
                target_batch=Variable(target_batch.cuda())
            prediction_batch = model(input_batch)

            _, _, H, W = prediction_batch.shape
            target_batch = torchvision.transforms.CenterCrop([H,W])(target_batch)

            loss = loss_function(prediction_batch,target_batch) #vergelijk predicted na image met de echte na image
            losses.append(loss.item())

            if plot == True and i%1 == 0:
                for j in range(self.batch_size):
                    fig = plt.subplots(figsize=(20,40))
                    img_pred = prediction_batch[j][0].cpu()
                    img_input = input_batch[j][0].cpu()
                    img_target = target_batch[j][0].cpu()
                    plt.subplot(3,1,1)
                    plt.imshow(img_input.detach().cpu().numpy(),cmap='viridis')
                    plt.title('Before injection')
                    plt.subplot(3,1,2)
                    plt.imshow(img_target.detach().numpy(),cmap='viridis')
                    plt.title('24 hours after injection with contrast enhancement')
                    plt.subplot(3,1,3)
                    plt.imshow(img_pred.detach().numpy(),cmap='viridis')
                    plt.title('Model prediction')
                    plt.savefig(f'afbeeldingen/{j}_{self.model_name}.png')
                    plt.close()
        av_loss = np.mean(np.array(losses))
        print(f'The average test loss is: {av_loss}')
        test_run = {'average_testloss':av_loss}
        
        with open(f'runlogs/{self.model_name}.json', 'w') as file:
                    json.dump(test_run, file, indent=4)
        return av_loss

