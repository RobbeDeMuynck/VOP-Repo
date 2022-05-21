from enhancement_load_data import MiceDataset, get_data
import torch
from torch.utils.data import DataLoader
from learning_train import train

### Declare device ###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

### Declare training hyperparameters ###
num_epochs = 300 # 300
batch_size = [4] # [4, 8, 12]
learning_rate = [1e-3] # [1e-2, 5e-3, 1e-3, 1e-4, 1e-5]
weight_decay = [0] # [0, 1e-2, 1e-4]
patience = 5 # 5

### Declare network architecture ###
layers = [3] # [3, 4]
features = [12] # [4, 8, 12, 16]

repeats = 1
### Train model ###
for LYRS in layers:
        for FT in features:
                for BS in batch_size:
                        for LR in learning_rate:
                                for WD in weight_decay:
                                        for i in range(repeats):
                                                ### Declare training & validation datasets ###
                                                input, target, val_input, val_target = get_data(plane='transversal', val_mouse=0)
                                                train_loader = DataLoader(MiceDataset(input, target), batch_size=BS, shuffle=True, drop_last=True)
                                                val_loader = DataLoader(MiceDataset(val_input, val_target), batch_size=BS, shuffle=True, drop_last=True)
                                                ### Train model ###
                                                model_name = 'learning'
                                                log_folder = 'runlogs_learning'
                                                if repeats > 1:
                                                        model_name += f';RUN={i}'
                                                        log_folder = 'runlogs_repeat'
                                                train(LYRS, FT, device,
                                                        train_loader, val_loader,
                                                        num_epochs, BS, learning_rate=LR, weight_decay=WD, patience=patience,
                                                        model_name=model_name, log_folder=log_folder, save=True)