import numpy as np
import torch
from torch.utils.data import Dataset

################################## INITIALIZING THE DATALOADER  ##################################
class MiceDataset(Dataset):
    def __init__(self, data_voor, data_na, p=0.5):
        super().__init__()
        self.data_voor = (data_voor - np.mean(data_voor))/np.std(data_voor) #vanwege de kleine dataset laden we het gewoon helemaal in memory en normaliseren we in place
        self.data_na = (data_na - np.mean(data_na))/np.std(data_na)
        self.p = p

    def __len__(self):
        return len(self.data_voor)

    def __getitem__(self, index):
        input = self.data_voor[index]
        target = self.data_na[index]
        
        ###  DATA AUGMENTATION: 50% chance of flipping the image (horizontally and/or vertically)
        if torch.rand(1) < self.p:
            input = np.flipud(input)
            target = np.flipud(target)

        if torch.rand(1) < self.p:
            input = np.fliplr(input)
            target = np.fliplr(target)
        
        input = torch.from_numpy(input.copy()).unsqueeze(0).float()
        target = torch.from_numpy(target.copy()).unsqueeze(0).float()
        return input, target