from _UNET import *
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from _data_loading import MiceDataset
import numpy as np
import tqdm
from torch.autograd import Variable


import json
import time

################################### DECLARING HYPERPARAMETERS  ##################################
# num_epochs = params.num_epochs
# batch_size = params.batch_size
# learning_rate = params.learning_rate
# weight_decay = params.weight_decay
# patience = params.patience
# features = params.features

################################### LOADING DATA TRANSVERSAL  ###################################
# input = MiceData.Train_coronal_001h
# target = MiceData.Train_coronal_024h
# val_input = MiceData.Test_coronal_001h
# val_target = MiceData.Test_coronal_024h

################################## TRAINING  ##################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
def TRAIN(input, target, val_input, val_target,
        num_epochs, batch_size, learning_rate, weight_decay, patience, features, layers,
        model_name='TEST', save=True):
    model = UNet(ft=features, layers=layers).to(device)
    optimizer = Adam(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                    )
    train_loader = DataLoader(
                    MiceDataset(input, target),
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True
                    )
    # test_loader = DataLoader(
    #                 MuizenDataset(?, ?),
    #                 batch_size=batch_size,
    #                 shuffle=True,
    #                 )
    val_loader = DataLoader(
                    MiceDataset(val_input, val_target),
                    batch_size=batch_size,
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
    for epoch in range(1, num_epochs+1):  # we itereren meerdere malen over de data tot convergence?
        model.train()
        train_epoch_loss = 0
        print(f"Epoch: {epoch}/{num_epochs}")
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

    # print(f"""Training_losses = {loss_stats["train"]}
    # Validation_losses = {loss_stats["val"]}""")

    # Dictionary with model details
    run = {
        'train_time' : endtime-starttime,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'patience': patience,
        'features': features,
        'train_loss': loss_stats["train"],
        'val_loss': loss_stats["val"], 
        'num_epoch_convergence': epoch_no
    }
    return run

##################################### Plotting MODEL losses  ##################################
# plt.semilogy(loss_stats["train"], label='Training losses')
# plt.semilogy(loss_stats["val"], label='Validation losses')
# plt.ylabel('MSE loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.grid()
# plt.show()