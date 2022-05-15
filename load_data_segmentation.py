import numpy as np
import torch
from torch.utils.data import Dataset
import pathlib
import nibabel as nib

import matplotlib.pyplot as plt

################################## DECLARING THE DATASET  ##################################
class MiceDataset(Dataset):
    def __init__(self, data_in, target, p=0.5):
        super().__init__()
        ### Normalize data ###
        self.data_in = (data_in - np.mean(data_in))/np.std(data_in)
        self.target = target
        self.p = p

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, index):
        input = self.data_in[index]
        target = self.target[index] # One-hot encoding!
        
        ###  DATA AUGMENTATION: 50% chance of flipping the image (horizontally and/or vertically)
        if torch.rand(1) < self.p:
            input = np.flipud(input)
            target = np.flipud(target)

        if torch.rand(1) < self.p:
            input = np.fliplr(input)
            target = np.fliplr(target)
        
        input = torch.from_numpy(input.copy()).unsqueeze(0).float()
        target = torch.from_numpy(target.copy()).float()
        return input, target

################################## GET PROCESSED DATA ##################################
def get_data(plane='sagittal', val_mice=[], test_mice=[]):
    train_input, train_target = [], []
    val_input, val_target = [], []
    test_input, test_target = [], []

    mice = ["M01", "M02","M03", "M04", "M05", "M06", "M07","M08","M09", "M10", "M11", "M12","M13", "M14", "M15", "M16", "M17","M18","M19", "M20"]
    path = pathlib.Path(__file__).parent

    for i, mouse in enumerate(mice):
        # Declare path to data of each mouse
        timestamp = "024h"

        path_ct = path / f"data_segmentation/{mouse}_{timestamp}/CT280.img"
        path_organ = path / f"data_segmentation/{mouse}_{timestamp}/Organ.img"
        path_class = path / f"data_segmentation/{mouse}_{timestamp}/Organ.cls"
        if not path_organ.is_file():
            path_organ = path / f"data_segmentation/{mouse}_{timestamp}/Organ1.img"
        if not path_class.is_file():
            path_class = path / f"data_segmentation/{mouse}_{timestamp}/Organ1.cls"
        
        ct = nib.load(path_ct).get_fdata()
        organ = nib.load(path_organ).get_fdata()
        L, H, W = ct.shape
        L_, H_, W_ = organ.shape
        print(f'shape of ct mouse {mouse}:\t({L}, {H}, {W})')
        print(f'shape of organ mouse {mouse}:\t({L_}, {H_}, {W_})\n')
        if plane == 'sagittal':
            ct, organ = ct, organ
            print(f'shape of ct mouse {mouse}:\t({ct.shape})')
        for j in range(100):
            if i+1 in val_mice:
                val_input.append(ct)
                val_target.append(organ)
            if i+1 in test_mice:
                test_input.append(ct)
                test_target.append(organ)
            if i+1 not in val_mice+test_mice:
                train_input.append(ct)
                train_target.append(organ)
        if i == 0:
            plt.imshow(ct[50,:,:], cmap='bone')
            plt.imshow(organ[50,:,:], cmap='viridis',alpha=.5)
            plt.show()

    train_input, train_target = np.array(train_input), np.array(train_target, dtype=int)
    val_input, val_target = np.array(val_input), np.array(val_target, dtype=int)
    test_input, test_target = np.array(test_input), np.array(test_target, dtype=int)

    return train_input, train_target, val_input, val_target, test_input, test_target

train_input, train_target, val_input, val_target, test_input, test_target = get_data(plane='sagittal', val_mice=[15, 16, 17], test_mice=[18, 19, 20])