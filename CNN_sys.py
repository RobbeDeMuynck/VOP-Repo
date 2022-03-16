# Import 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

