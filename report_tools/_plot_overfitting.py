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

with open(Path('./runlogs/LYRS={};FT={};BS={};LR={};WD={}.json'.format(3, 12, 8, 0.001, 0)), 'r') as RUN:
    run = json.load(RUN)
    losses = run["train_loss"], run["val_loss"]

n = len(losses)
# palette = sns.color_palette("mako_r", n)
# palette = sns.color_palette("crest", n)
palette = sns.color_palette("colorblind")
fs = 15
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.semilogy(losses[0], lw=5, color=palette[0], label='Training losses') # semilogy
ax.semilogy(losses[1], lw=5, color=palette[1], label='Validation losses')
ax.axvline(x=np.argmin(losses[1]), c='k', ls='--', label='Last saved version')
ax.set_title('MSE loss decay per epoch', fontsize=fs+10)
ax.set_xlabel('Epoch number', fontsize=fs+5)
ax.set_ylabel('MSE loss (batch-average)', fontsize=fs+5)
ax.legend(fontsize=fs)
ax.grid()

plt.tight_layout()
# plt.savefig('./IMAGES/Overfit_prevention.png', dpi=200)
plt.show()

