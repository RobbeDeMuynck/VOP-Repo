import seaborn as sns
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np

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
# 4, 12, 12, 0.001, 0

with open(Path('runlogs_overfitting/LYRS={};FT={};BS={};LR={};WD={}.json'.format(3, 12, 4, 0.001, 0)), 'r') as RUN:
    run = json.load(RUN)
    losses = run["train_loss"], run["val_loss"]

n = len(losses)
palette = sns.color_palette("mako_r", n)
palette = sns.color_palette("crest", n)
fig, ax = plt.subplots(1, 1)
ax.semilogy(losses[0], color=palette[0], label='Training losses')
ax.semilogy(losses[1], color=palette[1], label='Validation losses')
ax.axvline(x=np.argmin(losses[1]), c='k', ls='--', label='Last saved version')
ax.set_title('MSE loss decay per epoch')
ax.set_xlabel('Epoch number')
ax.set_ylabel('MSE loss')
ax.legend()
ax.grid()
plt.show()

