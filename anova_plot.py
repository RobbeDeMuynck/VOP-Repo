import numpy as np
import json
import seaborn as sns
import pandas as pd
 
f = open('anova.json')
 

data = json.load(f)
print(data)
losses = np.array(data['losses'])

tips = sns.load_dataset("tips")

ax = sns.boxplot(x="Validation Mouse", y="MSE loss", data=tips)
ax = sns.swarmplot(x="Validation Mouse", y="MSE loss", data=tips, color=".25")
