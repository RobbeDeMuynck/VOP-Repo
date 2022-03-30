from numpy import dtype
import seaborn as sns
import csv
import pandas as pd
import numpy as np
sns.set_theme(style="white")
import matplotlib.pyplot as plt
import json
from pathlib import Path




cols = ["Batch size","Learning rate","Weight decay","Starting features","Minimum validation loss","Epochs till convergence"]
vals = {col: [] for col in cols}
files = Path('runlogs').glob('*')
for file in files:
    with open(file, "r") as RUN:
        run = json.load(RUN)
        vals["Batch size"].append(run["batch_size"])
        vals["Learning rate"].append(run["learning_rate"])
        vals["Weight decay"].append(["weight_decay"])
        vals['Starting features'].append(run["features"])
        vals["Minimum validation loss"].append(min(run["val_loss"]))
        vals["Epochs till convergence"].append(run["num_epoch_convergence"])

Data = pd.DataFrame(data=vals,columns=cols)
print(Data.head())
sns.relplot(x="Epochs till convergence", y="Minimum validation loss", hue="Learning rate", style = "Starting features",size="Batch size",sizes=(40, 800), alpha=.5, palette="crest",height=6, data=Data)
plt.show()




# def MSE(input,output):
#     return np.mean((input.detach().cpu().numpy()-output.detach().cpu().numpy())**2)