import params
import MiceData
from UNET import *
from train import TRAIN
import json

#model_name = "model_test2"
################################### DECLARING HYPERPARAMETERS  ##################################
num_epochs = params.num_epochs
batch_size = params.batch_size
learning_rate = params.learning_rate
weight_decay = params.weight_decay
patience = params.patience

################################### LOADING DATA TRANSVERSAL  ###################################
input = MiceData.Train_transversal_001h
target = MiceData.Train_transversal_024h
val_input = MiceData.Test_transversal_001h
val_target = MiceData.Test_transversal_024h

################################## TRAINING  ##########################################

run = TRAIN(input, target, val_input, val_target,
        num_epochs, batch_size, learning_rate, weight_decay, patience,
        model_name='TEST', save=True)

##################################### WRITE MODEL IN RUNLOG   ##################################
run_name = 'TEST'
with open('runlog.json', 'w+') as file:
    data = {}
    data[run_name] = run
    json.dump(data, file, indent=4)