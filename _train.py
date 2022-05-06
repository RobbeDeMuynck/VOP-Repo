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

################################## TRAINING  ##################################
def train(layers, features, device,
        train_loader, val_loader,
        num_epochs, batch_size, learning_rate=1e-3, weight_decay=0, patience=5,
        model_name='TEST', log_folder='runlogs', save=True):

    ### Declare network architecture ###
    model = UNet(layers=layers, ft=features).to(device)
    ### Declare loss function & optimizer ###
    loss_function = nn.MSELoss().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print('Starting with training...')
    loss_stats = {
        'train': [],
        'val': []
        }
    best_loss = np.inf
    starttime = time.time()
    for epoch in range(1, num_epochs+1):
        model.train().to(device)
        train_epoch_loss = 0
        print(f"Epoch: {epoch}/{num_epochs}")
        for input_batch, target_batch in tqdm(train_loader):
            print(f'input:{input_batch.shape},target:{target_batch.shape}')
            # Put batch on GPU
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad()
            
            prediction_batch = model(input_batch)[0].to(device)
            _, _, H, W = prediction_batch.shape
            target_batch = torchvision.transforms.CenterCrop([H,W])(target_batch)
            loss = loss_function(prediction_batch, target_batch) # Compare prediction with target
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()
            
            input_batch = torchvision.transforms.CenterCrop([H,W])(input_batch)
        
        ### Prevent overfitting ###
        with torch.no_grad():
            val_epoch_loss = 0
                
            model.eval().to(device)
            for val_input_batch, val_target_batch in val_loader:
                val_input_batch = val_input_batch.to(device)
                val_target_batch = val_target_batch.to(device)

                val_pred = model(val_input_batch)[0].to(device)
                val_target_batch = torchvision.transforms.CenterCrop([H,W])(val_target_batch)

                val_loss = loss_function(val_pred, val_target_batch)
                val_epoch_loss += val_loss.item()
        
        # Store the batch-average MSE loss per epoch
        loss_stats["train"].append(train_epoch_loss/len(train_loader))
        loss_stats["val"].append(val_epoch_loss/len(val_loader))
        
        if loss_stats["val"][-1] < best_loss:
            endtime = time.time()
            best_loss, epoch_no = loss_stats["val"][-1], epoch
            if save == True:
                torch.save(model.state_dict(), 'MODELS/'+model_name+'.pth')
                print(f"    --> Model saved at epoch no. {epoch_no}")
        
        print(f"""
    Training loss: {loss_stats["train"][-1]}
    Last validation losses: {loss_stats['val'][-patience:]}
    Best loss: {best_loss}""")

        if np.all(np.array(loss_stats['val'][-patience:]) > best_loss):
            print(f"""
    Training terminated after epoch no. {epoch}
    Model saved as '{model_name}.pth': version at epoch no. {epoch_no}""")

            break
        elif epoch == num_epochs:
            epoch_no = epoch
            print(f"Model trained succesfully and saved as: '{model_name}.pth'")

    ### Dictionary with model details ###
    run = {
        'train_time' : endtime-starttime,
        'layers': layers,
        'features': features,

        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'patience': patience,
        
        'train_loss': loss_stats["train"],
        'val_loss': loss_stats["val"], 
        'num_epoch_convergence': epoch_no
    }
    with open(log_folder+f'/{model_name}.json', 'w+') as file:
                    json.dump(run, file, indent=4)
    return run

##################################### Plotting MODEL losses  ##################################
# plt.semilogy(loss_stats["train"], label='Training losses')
# plt.semilogy(loss_stats["val"], label='Validation losses')
# plt.ylabel('MSE loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.grid()
# plt.show()