import numpy as np
from matplotlib import pyplot as plt
import gluoncv.utils.viz.segmentation as seg

pallete = seg.vocpallete

label = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

left = np.array(range(20))
height = np.ones(20)

col = []
for i in range(1,21):
    col.append([x/255 for x in pallete[(i*3):(i*3+3)]])

plt.barh(left, height, tick_label=label, align="center",color=col)
plt.show()