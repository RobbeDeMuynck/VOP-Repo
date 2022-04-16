from numpy import dtype
import seaborn as sns
import csv
import pandas as pd
import numpy as np
sns.set_theme(style="white")
import matplotlib.pyplot as plt
import json
from pathlib import Path

files = Path('runlogs').glob('*')

cols = ["Layers","Starting features","Batch size","Learning rate","Weight decay","Minimum validation loss","Epochs trained","Training time [mins]"]
vals = {col: [] for col in cols}
for file in files:
    with open(file, "r") as RUN:
        run = json.load(RUN)
        if min(run["val_loss"]) < 0.006 and run["train_time"]/60 < 8:
            vals["Layers"].append(run["layers"])
            vals['Starting features'].append(run["features"])

            vals["Batch size"].append(run["batch_size"])
            vals["Learning rate"].append(run["learning_rate"])
            vals["Weight decay"].append(["weight_decay"])
            
            vals["Minimum validation loss"].append(min(run["val_loss"]))
            vals["Epochs trained"].append(run["num_epoch_convergence"])
            vals["Training time [mins]"].append(run["train_time"]/60)
        else:
            print(run["layers"], run["features"], run["batch_size"], run["learning_rate"], run["weight_decay"])
 
Data = pd.DataFrame(data=vals,columns=cols)
Data.head
print(Data.head())
sns.relplot(data=Data, x="Training time [mins]", y="Minimum validation loss", 
            hue="Learning rate", style = "Starting features", size="Batch size", col="Layers",
            sizes=(50, 350), alpha=.75, palette="colorblind", height=6)
plt.show()
# palette="crest"
# losses = []
# PAR = [(8, 0.001, 0.09, 4)]
# for file in files:
#    with open(file, "r") as RUN:
#        run = json.load(RUN)
#        params = run["batch_size"], run["learning_rate"], run["weight_decay"], run["features"]
#        train_loss = run["train_loss"]
#        val_loss = run["val_loss"]
#        # if params in PAR:
#        #     losses = train_loss, val_loss
#        #     print(run["num_epoch_convergence"])
#        losses = train_loss, val_loss

# with open(Path('runlogs/LYRS=4;FT=12;BS=12;LR=0.001;WD=0.json'), 'r') as RUN:
#     run = json.load(RUN)
#     losses = run["train_loss"], run["val_loss"]

# n = len(losses)
# palette = sns.color_palette("mako_r", n)
# palette = sns.color_palette("crest", n)
# fig, ax = plt.subplots(1, 1)
# ax.semilogy(losses[0], color=palette[0], label='Training losses')
# ax.semilogy(losses[1], color=palette[1], label='Validation losses')
# ax.axvline(x=np.argmin(losses[1]), c='k', ls='--', label='Last saved version')
# #f'BS=%d, LR=%.3f, WD=%.2f, FT=%d'%tuple(PAR[0])
# ax.set_title('MSE loss decay per epoch')
# ax.set_xlabel('Epoch number')
# ax.set_ylabel('MSE loss')
# ax.legend()
# ax.grid()
# plt.show()

# sns.lineplot(data=Data, y="Training", palette=palette)