import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# with open(path_class) as f:
        #     # EXAMPLE for "M03_024h/Organ_280.cls":
        #     # ClassColors=0 0 0 255|116 161 166 255|0 85 0 255|201 238 255 255|255 170 255 255|0 0 255 255|176 230 241 255|0 130 182 255|71 205 108 255|0 255 0 255|0 255 255 255|56 65 170 255|175 235 186 255
        #     # ClassIndices=0|1|2|3|4|5|6|7|8|9|10|11|12
        #     # ClassNames=unclassified|Trachea|Spleen|Bone|Lung|Heart|Stomach|Bladder|Muscle|Tumor|Kidneys|Liver|Intestine
ClassNames = {
        0: 'unclassified',
        1: 'Heart',
        2: 'Lung',
        3: 'Liver',
        4: 'Intestine',
        5: 'Spleen',
        6: 'Muscle',
        7: 'Stomach',
        8: 'Bladder',
        9: 'Bone',
        10: 'Kidneys',
        11: 'Trachea',
        12: 'Tumor'
}
ClassColors = {
    0: (0, 0, 0, 0),
    1: (116, 161, 166, 1),
    2: (0, 85, 0, 1),
    3: (201, 238, 255, 1),
    4: (255, 170, 255, 1),
    5: (0, 0, 255, 1),
    6: (176, 230, 241, 1),
    7: (0, 130, 182, 1),
    8: (71, 205, 108, 1),
    9: (0, 255, 0, 1),
    10: (0, 255, 255, 1),
    11: (56, 65, 170, 1),
    12: (175, 235, 186, 1)
}


def CM_plot(target, prediction, stats=True):
    '''target, prediction: 3D organs mask matrices (0-12)'''
    ### Check dimensions
    assert target.shape == prediction.shape, f'Shapes {target.shape} and {prediction.shape} do not match!'
    ### Calculate confusion 
    target, prediction = target.flatten(), prediction.flatten()
    cm = confusion_matrix(target, prediction, labels=[i for i in range(13)], normalize='true')

    if stats:
        accuracy = accuracy_score(target, prediction)
        # precision = precision_score(target, prediction, average='weighted')
        # recall = recall_score(target, precision, average='weighted')
        f1 = f1_score(target, prediction, average=None)
        print(f'''accuracy =\t{accuracy}\nf1-score = \t{f1}''')
        # precision =\t{precision}
        # recall =\t{recall}
        # calculated f1 =\t{2*precision*recall/(precision+recall)}

    ### Create dataframe and plot confusion matrix
    cm_df = pd.DataFrame(cm,
                        index = [ClassNames[idx] for idx in range(13)],
                        columns = [ClassNames[idx] for idx in range(13)])
    # print(cm_df.head())
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="white", font_scale = 1.25)
    g = sns.heatmap(cm_df, annot=True, fmt='.2f', cmap='Blues')
    # plot.set_title('Confusion Matrix')
    # plot.set_ylabel('Actual organ')
    # plot.set_xlabel('Predicted organ')
    plt.title('Confusion Matrix', fontsize=20, pad=10)
    plt.ylabel('Actual organ')
    plt.xlabel('Predicted organ', labelpad=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.savefig("./IMAGES/CM.png", dpi=200)
    plt.show()


# ### Generate plot
# sns.set_theme(style="white", font_scale = 1.25)

# N = 13
# RGBA = [(
#     ClassColors[index][0]/255,
#     ClassColors[index][1]/255,
#     ClassColors[index][2]/255,
#     ClassColors[index][-1]) for index in range(N)]
# mycm = matplotlib.colors.ListedColormap(RGBA)


# def plot_examples(colormap):
#     """
#     Help function to plot data with associated colormap.
#     """
#     np.random.seed(19680801)
#     data = np.random.randn(12, 12)
#     fig, axs = plt.subplots(1, 1, figsize=(1 * 2 + 2, 3), constrained_layout=True)
#     psm = axs.pcolormesh(data, cmap=colormap, rasterized=True)
#     fig.colorbar(psm, ax=axs)
#     plt.show()

# cmap = cm.get_cmap('Set3')
# RGBA = [(0, 0, 0, 0)]+[tuple(list(RGB)+[1]) for RGB in cmap.colors]
# cmap = matplotlib.colors.ListedColormap(RGBA)
# print(RGBA)

# plot_examples(cmap)

