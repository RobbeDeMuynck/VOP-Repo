import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

#### Legend for segmentation
cmap = cm.get_cmap('Set3')
RGBA = [(0, 0, 0, 0)]+[tuple(list(RGB)+[1]) for RGB in cmap.colors]
cmap = matplotlib.colors.ListedColormap(RGBA)
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
segmentation_legend_elements = [Patch(facecolor=rgba, label=ClassNames[i+1]) for i, rgba in enumerate(RGBA[1:])]

### Plotting functions
def segmentation_plot(ax, CT, organ_mask, legend=True):
    ### Define colormap
    cmap = cm.get_cmap('Set3')
    RGBA = [(0, 0, 0, 0)]+[tuple(list(RGB)+[1]) for RGB in cmap.colors]
    cmap = matplotlib.colors.ListedColormap(RGBA)

    ### Construct plot
    # fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
    ax.imshow(CT, cmap='bone')
    psm = ax.imshow(organ_mask, cmap=cmap, alpha=.425, vmin=-0.5, vmax=12.5)
    # fig.colorbar(psm, ax=axs)

    # Add legend patches
    if legend:
        ax.legend(handles=segmentation_legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    return None