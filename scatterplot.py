from numpy import dtype
import seaborn as sns
import csv
import pandas as pd
import numpy as np
sns.set_theme(style="white")
import matplotlib.pyplot as plt

with open("hyperparam.csv", "r") as f:
    # using csv.writer method from CSV package
    d = []
    for row in csv.reader(f, delimiter = ","):
        d.append(row)
d = [eval(i) for i in d[0]]
data = [[],[],[],[],[],[]]
for i in d:
    for j in range(len(i)):
        data[j].append(i[j])
kolommen = ["batch size","learning rate","weight decay","number of epochs","number of starting features","Loss"]
dj = {
    kolommen[k] : data[k] for k in range(len(data)-1)
}
dj[kolommen[-1]] = np.log10(np.array(data[-1]))

Data = pd.DataFrame(data=dj,columns=kolommen)
print(Data.head())
sns.relplot(x="batch size", y="Loss", hue="learning rate", size="number of epochs",sizes=(40, 400), alpha=.5, palette="crest",height=6, data=Data)
plt.show()
