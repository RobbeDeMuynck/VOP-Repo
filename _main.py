from _load_data import MiceDataset, get_data
import torch
from torch.utils.data import DataLoader
from _train import train

### Declare device ###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

### Declare training hyperparameters ###
num_epochs = 120
batch_size = 4
learning_rate = 1e-4
weight_decay = 0.9
patience = 5

### Declare training & validation datasets ###
input, target, val_input, val_target = get_data(plane='transversal', val_mouse=0)
train_loader = DataLoader(MiceDataset(input, target), batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(MiceDataset(val_input, val_target), batch_size=batch_size, shuffle=True, drop_last=True)

### Declare network architecture ###
features = 8
layers = 4

### Train model ###
train(layers, features, device,
        train_loader, val_loader,
        num_epochs, batch_size, learning_rate=learning_rate, weight_decay=weight_decay, patience=5,
        model_name='TEST2', save=True)