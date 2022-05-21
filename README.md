# Deep learning for reducing the quantity of contrast agents in microCT

The architechture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

***

## Overview

### Goal

The goal of this project is devided in 2 parts: contrast enhancement and organ segmentation on micro-CT images. For both methods a U-NET is implemented.

### Data

The dataset from the study of [S. Rosenhein et al.](https://www.nature.com/articles/sdata2018294) was used. This contained 6 mice (before and after contrast agent injection) that could be used for contrast enhancement and 16 mice (with organ labeling) that could be used for organ segmentation. This data was preprocessed before implementing.

### Model

A modified U-NET was implemented with the use of 'Pytorch'. This model can be altered for different number of:  Layers, Starting Features, Batch size and Learning rates.

![U-NET](../main/IMAGES/Unet.png)

### Training and testing

The model is trained until the validation loss has not decreased for 5 epochs. After training the model is optimized, using the Adam opimizer, for different sets of parameters: Layers, Starting Features, Batch size and Learning rates. This is done for both contrast enhancement and segmentation.

### Result
All results for contrast enhancement and organ segmentation can be found in the 'IMAGES' folder.

![Contrastt](../main/IMAGES/Result_bone_transversal1_M08_Layers=3,FT=16,BS=4,LR=0.001.png)

![Contrastc](../main/IMAGES/Result_bone_coronal_M08_Layers=3,FT=16,BS=4,LR=0.001.png)

![Contrasts](../main/IMAGES/Result_bone_sagittal_M08_Layers=3,FT=16,BS=4,LR=0.001.png)

![Segmentation](../main/IMAGES/segmentation_predict.png)

***

## How to use

### Overall
`convert_dicom.py` is used to convert the obatained image into Dicom format.

`preprocess.py` preprocesses the images by manually enhaning the contrast.

### Contrast enhancement

`enhancement_feature_extraction.py` returns a summary of the number of features and the size of the model.

`enhancement_load_data.py` loads the 3D images and converts it to transversal, sagittal and coronal slices.

`enhancement_main.py` trains the model for chosen parameters using `enhancement_train.py` which also saves the model after training.

`enhancement_molecubes.py` applies a model to self obtained scans for the Infinity lab at UZ Ghent. The resulting images are plotted and saved.

`enhancement_predict.py` applies a model to scans from the dataset. The resulting images are plotted and saved.

`enhancement_UNET.py` contains the architecture of the modified U-NET for the contrast enhancement task.

`learning_main.py` does the same as `enhancement_main.py`.

`learning_main.py` does the same as `enhancement_main.py` but uses `learning_train.py` to train.

### Segmentation

`segmentation_load_data.py` loads the data and creates a correct labeling of the organs. It return the loaded data as sagittal slices.

`segmentation_performance.py` creates a confusion matrix to evaluate the performance of the model.

`segmentation_predict` applies a model to scans from the dataset. The resulting images with the predicted organ masks are plotted and saved.

`segmentation_train` trains the model for chosen parameters and saves the trained model.

`segmentation_UNET` contains the architecture of the modified U-NET for the segmentation task.
