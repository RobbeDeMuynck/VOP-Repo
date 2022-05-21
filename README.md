# Deep learning for reducing the quantity of contrast agents in microCT

The architechture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

***

## Overview

### Goal

The goal of this project is devided in 2 parts: contrast enhancement and organ segmentation on micro-CT images. For both methods a U-NET is implemented.

### Data

The dataset from the study of [S. Rosenhein et al.](https://www.nature.com/articles/sdata2018294) was used. This contained 6 mice that could be used for contrast enhancement and 16 that could be used for organ segmentation. This data was preprocessed before implementing.

### Model

A modified U-NET was implemented with the use of 'Pytorch'. This model can be altered for different number of:  Layers, Starting Features, Batch size and Learning rates.

![U-NET](IMAGES\Unet.png)

### Training and testing

The model is trained until the validation loss has not decreased for 5 epochs. After training the model is optimized, using the Adam opimizer, for different sets of parameters: Layers, Starting Features, Batch size and Learning rates. This is done for both contrast enhancement and segmentation.

### Result
Results for contrast enhancement:

![Contrast](.\IMAGES\Result_bone_transversal1_M08_Layers=3,FT=16,BS=4,LR=0.001.png)

![Contrast](IMAGES\Result_bone_coronal_M08_Layers=3,FT=16,BS=4,LR=0.001.png)

![Contrast](IMAGES\Result_bone_sagittal_M08_Layers=3,FT=16,BS=4,LR=0.001.png)

***

## How to use




