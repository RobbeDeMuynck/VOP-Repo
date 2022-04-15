import numpy as np
import torch
from torch.utils.data import Dataset
import pathlib
import nibabel as nib

################################## DECLARING THE DATASET  ##################################
class MiceDataset(Dataset):
    def __init__(self, data_in, data_out, p=0.5):
        super().__init__()
        ### Normalize data ###
        self.data_in = (data_in - np.mean(data_in))/np.std(data_in)
        self.data_out = (data_out - np.mean(data_out))/np.std(data_out)
        self.p = p

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, index):
        input = self.data_in[index]
        target = self.data_out[index]
        
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

################################## GET PROCESSED DATA ##################################
def get_data(plane='transversal', val_mouse=6): 
        train_input, train_target = [], []
        val_input, val_target = [], []

        mice = ["M03", "M04", "M05", "M06", "M07","M08"]
        train_names = [mouse for i, mouse in enumerate(mice) if i!= val_mouse]
        val_names = [mice[val_mouse]]

        path = pathlib.Path('processed').parent
        for timestamp in ["-001h", "024h"]:
            for mouse in train_names:
                if timestamp == "-001h":
                    path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
                    train_input.append(nib.load(path_ct).get_fdata())
                else: 
                    path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
                    train_target.append(nib.load(path_ct).get_fdata())
            for mouse in val_names:
                if timestamp == "-001h":
                    path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
                    val_input.append(nib.load(path_ct).get_fdata())
                else: 
                    path_ct = path / f"processed/{mouse}_{timestamp}_CT280.img"
                    val_target.append(nib.load(path_ct).get_fdata())

        ### Return transversal slices ###
        if plane == 'transverse':

            train_transversal_001h = []
            train_transversal_024h = []
            for mouse in train_input:
                for i in range(mouse.shape[-1]):
                    train_transversal_001h.append(mouse[:,:,i])
            for mouse in train_target:
                for i in range(mouse.shape[-1]):
                    train_transversal_024h.append(mouse[:,:,i])

            val_transversal_001h = []
            val_transversal_024h = []
            for mouse in val_input:
                for i in range(mouse.shape[-1]):
                    val_transversal_001h.append(mouse[:,:,i])
            for mouse in val_target:
                for i in range(mouse.shape[-1]):
                    val_transversal_024h.append(mouse[:,:,i])
            
            print('Data successfully initialized')
            return train_transversal_001h, train_transversal_024h, val_transversal_001h, val_transversal_024h   
            # train_transversal_001h (1210 slices)
            # train_transversal_024h (1210 slices)
            # val_transversal_001h  (242 slices)
            # val_transversal_024h  (242 slices)

        ### Loading sagittal slices ###
        elif plane=='sagittal':

            train_sagittal_001h = []
            train_sagittal_024h = []
            for mouse in train_input:
                for i in range(mouse.shape[1]):
                    train_sagittal_001h.append(mouse[:,i,:])
            for mouse in train_target:
                for i in range(mouse.shape[1]):
                    train_sagittal_024h.append(mouse[:,i,:])

            val_sagittal_001h = []
            val_sagittal_024h = []
            for mouse in val_input:
                for i in range(mouse.shape[1]):
                    val_sagittal_001h.append(mouse[:,i,:])
            for mouse in val_target:
                for i in range(mouse.shape[1]):
                    val_sagittal_024h.append(mouse[:,i,:])

            print('Data successfully initialized')
            return train_sagittal_001h, train_sagittal_024h, val_sagittal_001h, val_sagittal_024h
            #Train_sagittal_001h (500 slices)
            #Train_sagittal_024h (500 slices)
            #val_sagittal_001h  (100 slices)
            #val_sagittal_024h  (100 slices)

        ### Loading coronal slices ###
        elif plane=='coronal':

            train_coronal_001h = []
            train_coronal_024h = []
            for mouse in train_input:
                for i in range(mouse.shape[0]):
                    train_coronal_001h.append(mouse[i,:,:])
            for mouse in train_target:
                for i in range(mouse.shape[0]):
                    train_coronal_024h.append(mouse[i,:,:])

            val_coronal_001h = []
            val_coronal_024h = []
            for mouse in val_input:
                for i in range(mouse.shape[0]):
                    val_coronal_001h.append(mouse[i,:,:])
            for mouse in train_target:
                for i in range(mouse.shape[0]):
                    val_coronal_024h.append(mouse[i,:,:])
            print('Data successfully initialized')
            return train_coronal_001h, train_coronal_024h, val_coronal_001h, val_coronal_024h

        print('Data loading failed')
        return None