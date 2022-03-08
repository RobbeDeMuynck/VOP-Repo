import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train = Import_data("C:\\Users\\Werk\\OneDrive - UGent\\Burgie 4\\Sem 2\\VOP\\Coding\\microCT data X-Cube\\584_l0r0\\20211021150245_CT_ISRA_0.dcm")
test = Import_data("C:\\Users\\Werk\\OneDrive - UGent\\Burgie 4\\Sem 2\\VOP\\Coding\\microCT data X-Cube\\584_l0r0\\20211021150245_CT_ISRA_0.dcm")

trainset = torch.utils.data.DataLoader(train,batch_size=10,shuffle=True)
testset = torch.utils.data.DataLoader(test,batch_size=10,shuffle=False)

class Import_data():
  def __init__(self, filename_train, filename_test):
    train = filename_train
    test = filename_test
    normalized_train = torch.from_numpy(self.normaliseer(train).astype(np.float32))
    normalized_test = torch.from_numpy(self.normaliseer(test).astype(np.float32))
    maxpooled_train = self.maxpool(normalized_train)
    maxpooled_test = self.maxpool(normalized_test)

  def softmax(self,x):
    return np.exp(x)/np.sum(np.exp(x))

  def normaliseer(self,x):
    return (x-np.mean(x))/np.std(x)
  
  def maxpool(self,x):
    m = torch.nn.MaxPool2d(5,stride = 3)
    return m(x)
  
  def get_traindata(self):
    return self.maxpooled_train.view(-1,40000)

  def get_testdata(self):
    return self.maxpooled_test.view(-1,40000)