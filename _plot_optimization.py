import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np

files = Path('runlogs').glob('*')

cols = ["Layers","Starting features","Batch size","Learning rate","Weight decay","Minimum validation loss","Epochs trained","Training time [mins]"]
vals = {col: [] for col in cols}

train_time = []
for file in files:
    with open(file, "r") as RUN:
        run = json.load(RUN)
        ### Select cases using 'if'-statement ###
        train_time.append(run["train_time"])
        if True: #min(run["val_loss"]) < 0.01 and run["train_time"]/60 < 8: # and run["learning_rate"] >= 0.001
            vals["Layers"].append(run["layers"])
            vals['Starting features'].append(run["features"])

            vals["Batch size"].append(run["batch_size"])
            vals["Learning rate"].append(run["learning_rate"])
            vals["Weight decay"].append(["weight_decay"])
            
            vals["Minimum validation loss"].append(min(run["val_loss"]))
            vals["Epochs trained"].append(run["num_epoch_convergence"])
            vals["Training time [mins]"].append(run["train_time"]/60)

            ### PLOT TUNING REPEAT
            # if 'RUN' not in file.name:
            #     vals["Minimum validation loss"][-1] += 1

        else:
            print(run["layers"], run["features"], run["batch_size"], run["learning_rate"], run["weight_decay"])
 
Data = pd.DataFrame(data=vals,columns=cols)

print(f"""
Number of networks trained:\t{len(train_time)}
Total training time [mins]:\t{sum(train_time)/60} 
Average training time:\t{np.mean(train_time)/60}""")

sns.set_theme(style="white", font_scale = 1.25)
sns.relplot(data=Data, x="Training time [mins]", y="Minimum validation loss", 
            hue="Learning rate", style = "Starting features", size="Batch size", col="Layers",
            sizes=(50, 350), alpha=.75, palette="colorblind", height=6)#.set(ylabel="Validation MSE loss (batch-average)").set(xlim=(0, 2.2),ylim=(0.0025,0.0055), ylabel="Validation MSE loss (batch-average)")

# plt.savefig("IMAGES/TUNING3.png", dpi=200)
plt.show()

# sns.lineplot(data=Data, y="Training", palette=palette)