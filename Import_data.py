import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import imageio

train = Import_data("C:\\Users\\Werk\\OneDrive - UGent\\Burgie 4\\Sem 2\\VOP\\Coding\\microCT data X-Cube\\584_l0r0\\20211021150245_CT_ISRA_0.dcm")
test = Import_data("C:\\Users\\Werk\\OneDrive - UGent\\Burgie 4\\Sem 2\\VOP\\Coding\\microCT data X-Cube\\584_l0r0\\20211021150245_CT_ISRA_0.dcm")

#load and shuffle data
trainset = torch.utils.data.DataLoader(train,batch_size=10,shuffle=True)
testset = torch.utils.data.DataLoader(test,batch_size=10,shuffle=False)

class Import_data():
  def __init__(self, filename_train, filename_test):
    self.train, self.test = None, None
    
  def softmax(self,x):
    return np.exp(x)/np.sum(np.exp(x))

  def normalize(self):
    self.test = (self.test-np.mean(self.test))/np.std(self.test)
    self.train = (self.train-np.mean(self.train))/np.std(self.train)
    return None
  
  def maxpool(self,x):
    m = torch.nn.MaxPool2d(5,stride = 3)
    return m(x)
  
  def get_traindata(self):
    return self.train

  def get_testdata(self):
    return self.test

  def arrange_traindata(self):
    list_traindata = []
    for i in range(len(self.train)):
      list_traindata.append(self.train[4*i:4*i+4])
    return list_traindata

  def arrange_testdata(self):
    list_testdata = []
    for i in range(len(self.test)):
      list_testdata.append(self.test[4*i:4*i+4])
    return list_testdata

  def load_data(self,filename_train, filename_test):
    imageio.plugins.freeimage.download()
    img = imageio.imread(hdr_path, format='HDR-FI')
    return None
  
  def shape_data(self):
    

    return None

  def run(self):
    self.load_data()
    self.normalize()
    self.arrange_testdata()
    self.arrange_traindata()
    self.shape_data()
