import numpy as np
import torch
from torch.utils.data import Dataset
import pathlib
import nibabel as nib

import matplotlib.pyplot as plt

def mapping(old_organ, dict):
    # old_organ: 3D
    I, J, K = old_organ.shape
    new_organ = old_organ.copy()
    for i in range(I):
        for j in range(J):
            for k in range(K):
                new_organ[i, j, k] = dict[old_organ[i, j, k]]
    return new_organ


def get_data(plane='sagittal', val_mice=[], test_mice=[]):
    train_input, train_target = [], []
    val_input, val_target = [], []
    test_input, test_target = [], []

    mice = ["M01", "M02","M03", "M04", "M05", "M06", "M07","M08","M09", "M10", "M11", "M12","M13", "M14", "M15", "M16", "M17","M18","M19", "M20"]
    path = pathlib.Path(__file__).parent

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

        if i == 7:
            with open(path_class) as f:
                content = f.read()
                indices = content.split('\n')[1].split('=')[1].split('|')
                names = content.split('\n')[2].split('=')[1].split('|')
                dict = {int(idx): name for idx, name in zip(indices, names)}
            
                mapping_old_to_standard = {}
                for old_idx, old_name in dict.items():
                    mapping_old_to_standard[old_idx] = name_to_idx_standard[old_name]
                print(mapping_old_to_standard)
                organ = mapping(organ, mapping_old_to_standard)

        L, H, W = ct.shape
        L_, H_, W_ = organ.shape
        # print(f'shape of ct mouse {mouse}:\t({L}, {H}, {W})')
        # print(f'shape of organ mouse {mouse}:\t({L_}, {H_}, {W_})\n')
        if plane == 'sagittal':
            ct, organ = ct, organ
            # print(f'shape of ct mouse {mouse}:\t({ct.shape})')
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
        # if i == 0:
        #     plt.imshow(ct[50,:,:], cmap='bone')
        #     plt.imshow(organ[50,:,:], cmap='viridis',alpha=.5)
        #     plt.show()

    # train_input, train_target = np.array(train_input), np.array(train_target, dtype=int)
    # val_input, val_target = np.array(val_input), np.array(val_target, dtype=int)
    # test_input, test_target = np.array(test_input), np.array(test_target, dtype=int)
    return None
    # return train_input, train_target, val_input, val_target, test_input, test_target

# train_input, train_target, val_input, val_target, test_input, test_target = get_data(plane='sagittal', val_mice=[15, 16, 17], test_mice=[18, 19, 20])

get_data(plane='sagittal', val_mice=[15, 16, 17], test_mice=[18, 19, 20])