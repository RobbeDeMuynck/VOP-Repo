import params
import MiceData
from UNET import *
from train import TRAIN
import json

#model_name = "model_test2"
################################### DECLARING HYPERPARAMETERS  ##################################
# num_epochs = params.num_epochs
# batch_size = params.batch_size
# learning_rate = params.learning_rate
# weight_decay = params.weight_decay
# patience = params.patience
# features = params.features

num_epochs = 120
batch_sizes = [8]
learning_rates = [.001]
weight_decay = 0.09
patience = 5
features = [16]

################################## k-fold crossvalidation (k=6)  ##########################################
for k in range(6):
    input,target,val_input,val_target = MiceData.prep_data(test_mouse=k)
    WD = weight_decay
    for BS in batch_sizes:
        for LR in learning_rates:
            for FT in features:
                model_name = f'BS={BS};LR={LR};WD={WD};FT={FT};test_mouse={k}'
                run = TRAIN(input, target, val_input, val_target,
                        num_epochs, BS, LR, weight_decay, patience, FT,
                        model_name=model_name, save=True)
                # data[model_name] = run
                with open(f'runlogs_kfold/{model_name}.json', 'w+') as file:
                    json.dump(run, file, indent=4)
################################### LOADING DATA CORONAL  ###################################







################################## TRAINING  ##########################################
############################# WRITE MODEL IN RUNLOG   ##################################
# data = {}
# WD = weight_decay
# for BS in batch_sizes:
#     for LR in learning_rates:
#         for FT in features:
#             model_name = f'BS={BS};LR={LR};WD={WD};FT={FT}'
#             run = TRAIN(input, target, val_input, val_target,
#                     num_epochs, BS, LR, weight_decay, patience, FT,
#                     model_name=model_name, save=True)
#             # data[model_name] = run
#             with open(f'runlogs/{model_name}.json', 'w+') as file:
#                 json.dump(run, file, indent=4)


