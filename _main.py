from _load_data import MiceDataset, get_data
import torch
from torch.utils.data import DataLoader
from _train import train

from torch.optim import Adam
from torch.autograd import Variable
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

### Declare device ###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

### Declare training hyperparameters ###
num_epochs = 120
batch_size = 4
learning_rate = 1e-3
weight_decay = 0
patience = 5

### Declare training & validation datasets ###
input, target, val_input, val_target = get_data(plane='transverse', val_mouse=5)
train_loader = DataLoader(MiceDataset(input, target), batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(MiceDataset(val_input, val_target), batch_size=batch_size, shuffle=True, drop_last=True)

### Declare network architecture ###
features = 10
layers = 3

### Train model ###
train(features, layers, device,
        train_loader, val_loader,
        num_epochs, batch_size, learning_rate=1e-3, weight_decay=0, patience=5,
        model_name='TEST', save=True)