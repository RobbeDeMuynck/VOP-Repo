import seaborn as sns
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np
import pandas as pd

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

with open(Path('./runlogs_learning/learning.json'), 'r') as RUN:
    run = json.load(RUN)
    losses = run["pred_target"], run["input_target"]

window_size = 50

numbers_series = pd.Series(losses[0]), pd.Series(losses[1])
windows = numbers_series[0].rolling(window_size), numbers_series[1].rolling(window_size)
moving_averages = windows[0].mean(), windows[1].mean()

moving_averages_list = moving_averages[0].tolist(), moving_averages[1].tolist()

#without_nans = moving_averages_list[window_size - 1:]
losses = moving_averages_list



n = len(losses)
# palette = sns.color_palette("mako_r", n)
# palette = sns.color_palette("crest", n)
palette = sns.color_palette("colorblind")
fs = 15
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.semilogy(losses[1], lw=2, color=palette[1], label='Input - Target')
ax.semilogy(losses[0], lw=2, color=palette[0], label='Prediction - Target') # semilogy
#ax.axvline(x=np.argmin(losses[1]), c='k', ls='--', label='Last saved version')
ax.set_title('MSE per batch', fontsize=fs+10)
ax.set_xlabel('Batch number', fontsize=fs+5)
ax.set_ylabel('MSE loss', fontsize=fs+5)
ax.legend(fontsize=fs)
ax.grid()

plt.tight_layout()
# plt.savefig('./IMAGES/Learning.png', dpi=200)
plt.show()
