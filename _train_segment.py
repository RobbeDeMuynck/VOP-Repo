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
        model_name='segment', save=True):

    ### Declare network architecture ###
    model = UNet(layers=layers, ft=features).to(device)
    ### Declare loss function & optimizer ###
    loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device=device))
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(1, num_epochs+1):
        model.train().to(device)
        train_epoch_loss = 0
        print(f"Epoch: {epoch}/{num_epochs}")
        for input_batch, target_batch in tqdm(train_loader):
            
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
    print(train_epoch_loss)
    return train_epoch_loss

##################################### Plotting MODEL losses  ##################################
# plt.semilogy(loss_stats["train"], label='Training losses')
# plt.semilogy(loss_stats["val"], label='Validation losses')
# plt.ylabel('MSE loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.grid()
# plt.show()