import numpy as np
import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg

data = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/PlantGrowth.csv"

losses = []
for m in range(6):
    for j in range(10):
        path = f'runlogs_kfold/LYRS=3;FT=12;BS=4;LR=0.001;WD=0,valm={m};RUN={j}.json'
        f = open(path)
        data = json.load(f)
        losses.append(data['val_loss'][-1])

losses = np.array(losses)

ind = []
for i in range(6):
    for j in range(10):
        ind.append(i)
data = {'Mouse': ind,'MSE losses' : losses}
df=pd.DataFrame(data)

aov = pg.anova(data=df, dv='MSE losses', between='Mouse', detailed=True)
print(aov)

print(df.head())
ax = sns.boxplot(x="Mouse", y="MSE losses", data=df,palette='rocket')
ax = sns.swarmplot(x="Mouse", y="MSE losses", data=df, color=".1",alpha=.6)
plt.title('6-fold crossvalidation')
plt.show()