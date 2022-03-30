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
# cols = ["Batch size","Learning rate","Weight decay","Starting features","Minimum validation loss","Epochs till convergence"]
# vals = {col: [] for col in cols}
# for file in files:
#     with open(file, "r") as RUN:
#         run = json.load(RUN)
#         vals["Batch size"].append(run["batch_size"])
#         vals["Learning rate"].append(run["learning_rate"])
#         vals["Weight decay"].append(["weight_decay"])
#         vals['Starting features'].append(run["features"])
#         vals["Minimum validation loss"].append(min(run["val_loss"]))
#         vals["Epochs till convergence"].append(run["num_epoch_convergence"])

# Data = pd.DataFrame(data=vals,columns=cols)
# print(Data.head())
# sns.relplot(data=Data, x="Epochs till convergence", y="Minimum validation loss", 
#             hue="Learning rate", style = "Starting features", size="Batch size",
#             sizes=(50, 350), alpha=.5, palette="crest", height=6)
# plt.show()

losses = []
for file in files:
    with open(file, "r") as RUN:
        run = json.load(RUN)
        params = run["batch_size"], run["learning_rate"], run["weight_decay"], run["features"]
        train_loss = run["train_loss"]
        losses.append([params, train_loss])

n = len(losses)
palette = sns.color_palette("mako_r", n)
fig, ax = plt.subplots(1, 1)
for i, loss in enumerate(losses):
    BS, LR, WD, FT = tuple(loss[0])
    train_loss = loss[1]
    ax.plot(train_loss, color=palette[i], label=f'BS={BS};LR={LR};WD={WD};FT={FT}')
ax.set_xlabel('Epoch number')
ax.set_ylabel('Training losses')
ax.legend()
plt.show()

# sns.lineplot(data=Data, y="Training", palette=palette)