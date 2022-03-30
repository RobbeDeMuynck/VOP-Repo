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
batch_sizes = [4, 8, 12]
learning_rates = [0.01, 0.005, .001]
weight_decay = 0.09
patience = 5
features = [4, 10, 16]


################################### LOADING DATA TRANSVERSAL  ###################################
input = MiceData.Train_transversal_001h
target = MiceData.Train_transversal_024h
val_input = MiceData.Test_transversal_001h
val_target = MiceData.Test_transversal_024h

################################## TRAINING  ##########################################
############################# WRITE MODEL IN RUNLOG   ##################################
# data = {}
WD = weight_decay
for BS in batch_sizes:
    for LR in learning_rates:
        for FT in features:
            model_name = f'BS={BS};LR={LR};WD={WD};FT={FT}'
            run = TRAIN(input, target, val_input, val_target,
                    num_epochs, BS, LR, weight_decay, patience, FT,
                    model_name=model_name, save=True)
            # data[model_name] = run
            with open(f'/runlogs/{model_name}.json', 'w+') as file:
                json.dump(run, file, indent=4)


