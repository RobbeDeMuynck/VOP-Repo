import numpy as np
import torch
from torch.utils.data import Dataset
import pathlib
import nibabel as nib
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

# from report_tools.confusion_matrix import ClassNames

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
def mapping(old_organ, dict):
    # old_organ: 3D
    I, J, K = old_organ.shape
    new_organ = np.zeros((I, J, K))
    for i in range(I):
        for j in range(J):
            for k in range(K):
                new_organ[i, j, k] = dict[old_organ[i, j, k]]
    return new_organ

def get_data(plane='sagittal', val_mice=[], test_mice=[]):
    # Construct data lists
    train_input, train_target = [], []
    val_input, val_target = [], []
    test_input, test_target = [], []

    mice = ["M01", "M02","M03", "M04", "M05", "M06", "M07","M08","M09", "M10", "M11", "M12","M13", "M14", "M15", "M16", "M17","M18","M19", "M20"]
    path = pathlib.Path(__file__).parent

    # Standardized index of organ segmentation: dictionaries
    ClassNames = {
        0: 'unclassified',
        1: 'Heart',
        2: 'Lung',
        3: 'Liver',
        4: 'Intestine',
        5: 'Spleen',
        6: 'Muscle',
        7: 'Stomach',
        8: 'Bladder',
        9: 'Bone',
        10: 'Kidneys',
        11: 'Trachea',
        12: 'Tumor'
    }
    name_to_idx_standard = {val: key for key, val in ClassNames.items()}

    # Add each mouse
    for i, mouse in tqdm(enumerate(mice)):
        # Declare path to data of each mouse
        timestamp = "024h"

        path_ct = path / f"data_segmentation/{mouse}_{timestamp}/CT280.img"
        path_organ = path / f"data_segmentation/{mouse}_{timestamp}/Organ.img"
        path_class = path / f"data_segmentation/{mouse}_{timestamp}/Organ.cls"
        if not path_organ.is_file():
            path_organ = path / f"data_segmentation/{mouse}_{timestamp}/Organ1.img"
        if not path_class.is_file():
            path_class = path / f"data_segmentation/{mouse}_{timestamp}/Organ1.cls"
        
        # Read-in data
        ct = nib.load(path_ct).get_fdata()
        organ = nib.load(path_organ).get_fdata()

        # L, H, W = ct.shape
        # L_, H_, W_ = organ.shape
        # print(f'shape of ct mouse {mouse}:\t({L}, {H}, {W})')
        # print(f'shape of organ mouse {mouse}:\t({L_}, {H_}, {W_})')

        # Crop all scans to same dimensions
        ct, organ = ct[:140, :102, :190], organ[:140, :102, :190]

        # if plane == 'sagittal':
        #     ct, organ = ct[:140, :102, :194], organ[:140, :102, :194]

        # Standardized index of organ segmentation
        with open(path_class) as f:
            content = f.read()
            indices = content.split('\n')[1].split('=')[1].split('|')
            names = content.split('\n')[2].split('=')[1].split('|')
            dict = {int(idx): name for idx, name in zip(indices, names)}
        
            mapping_old_to_standard = {}
            for old_idx, old_name in dict.items():
                mapping_old_to_standard[old_idx] = name_to_idx_standard[old_name]
            
            organ = mapping(organ, mapping_old_to_standard)
            print(mapping_old_to_standard)

        # Append each mouse to
        if i+1 in val_mice:
            # print(ct[0].shape)
            val_input.extend([ct_slice for ct_slice in ct])
            val_target.extend([organ_slice for organ_slice in organ])
        if i+1 in test_mice:
            test_input.extend([ct_slice for ct_slice in ct])
            test_target.extend([organ_slice for organ_slice in organ])
        if i+1 not in val_mice+test_mice:
            train_input.extend([ct_slice for ct_slice in ct])
            train_target.extend([organ_slice for organ_slice in organ])

        # if i == 0:
        #     plt.imshow(ct[50,:,:], cmap='bone')
        #     plt.imshow(organ[50,:,:], cmap='viridis',alpha=.5)
        #     plt.show()
    # print(train_target)

    # Convert data lists to arrays
    train_input, train_target = np.array(train_input), np.array(train_target, dtype=int)
    val_input, val_target = np.array(val_input), np.array(val_target, dtype=int)
    test_input, test_target = np.array(test_input), np.array(test_target, dtype=int)

    return train_input, train_target, val_input, val_target, test_input, test_target

train_input, train_target, val_input, val_target, test_input, test_target = get_data(plane='sagittal', val_mice=[15, 16, 17], test_mice=[18, 19, 20])

# PLOT
idx = 140+70
ct, organ = val_input[idx], val_target[idx] 


cmap = cm.get_cmap('Set3')
RGBA = [(0, 0, 0, 0)]+[tuple(list(RGB)+[1]) for RGB in cmap.colors]
cmap = matplotlib.colors.ListedColormap(RGBA)

fig, axs = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
axs.imshow(ct, cmap='bone')
psm = axs.imshow(organ, cmap=cmap, alpha=.5, vmin=-0.5, vmax=12.5)
fig.colorbar(psm, ax=axs)

# Add legend patches
ClassNames = {
        0: 'unclassified',
        1: 'Heart',
        2: 'Lung',
        3: 'Liver',
        4: 'Intestine',
        5: 'Spleen',
        6: 'Muscle',
        7: 'Stomach',
        8: 'Bladder',
        9: 'Bone',
        10: 'Kidneys',
        11: 'Trachea',
        12: 'Tumor'
    }
legend_elements = [Patch(facecolor=rgba, label=ClassNames[i+1]) for i, rgba in enumerate(RGBA[1:])]
axs.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)
plt.show()
