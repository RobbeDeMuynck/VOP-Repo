from _load_data import MiceDataset, get_data
import torch
from torch.utils.data import DataLoader
from _train import train

### Declare device ###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

### Declare training hyperparameters ###
num_epochs = 300
batch_size = [4, 8, 12]
learning_rate = [1e-2, 5e-3, 1e-3, 1e-4, 1e-5]
weight_decay = [0] # [0, 1e-2, 1e-4]
patience = 5

### Declare network architecture ###
layers = [3, 4]
features = [16] # [4, 8, 12, 16]


### Train model ###
for LYRS in layers:
        for FT in features:
                for BS in batch_size:
                        for LR in learning_rate:
                                for WD in weight_decay:
                                        ### Declare training & validation datasets ###
                                        input, target, val_input, val_target = get_data(plane='transversal', val_mouse=0)
                                        train_loader = DataLoader(MiceDataset(input, target), batch_size=BS, shuffle=True, drop_last=True)
                                        val_loader = DataLoader(MiceDataset(val_input, val_target), batch_size=BS, shuffle=True, drop_last=True)
                                        ### Train model ###
                                        model_name = f'LYRS={LYRS};FT={FT};BS={BS};LR={LR};WD={WD}'
                                        train(LYRS, FT, device,
                                                train_loader, val_loader,
                                                num_epochs, BS, learning_rate=LR, weight_decay=WD, patience=5,
                                                model_name=model_name, save=True)