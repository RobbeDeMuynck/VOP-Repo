import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

  def normalize(self):
    for i in range(len(self.test)):
      self.test[i] = (self.test[i]-np.mean(self.test[i]))/np.std(self.test[i])
    for i in range(len(self.train)):
      self.train[i] = (self.train[i]-np.mean(self.train[i]))/np.std(self.train[i])
    return None
  
  def maxpool(self,x):
    m = torch.nn.MaxPool2d(5,stride = 3)
    return m(x)
  
  def get_traindata(self):
    return self.maxpooled_train.view(-1,40000)

  def get_testdata(self):
    return self.maxpooled_test.view(-1,40000)

  def arrange_traindata(self):
    list_traindata = []
    for i in range(len(self.train)):
      list_traindata.append(self.train[4*i:4*i+4])
    self.train = list_traindata
    return None

  def arrange_testdata(self):
    list_testdata = []
    for i in range(len(self.test)):
      list_testdata.append(self.test[4*i:4*i+4])
    self.test = list_testdata
    return None

  def load_data(self,filename_train, filename_test):
    imageio.plugins.freeimage.download()
    img = imageio.imread(hdr_path, format='HDR-FI')
    return None
  
  def shape_data(self):
    self.train = self.train.view[-1,40000]
    self.test = self.test.view[-1,40000]
    return None

  def run(self):
    self.load_data()
    self.normalize()
    self.arrange_testdata()
    self.arrange_traindata()
    self.shape_data()


train = Import_data("C:\\Users\\Werk\\OneDrive - UGent\\Burgie 4\\Sem 2\\VOP\\Coding\\microCT data X-Cube\\584_l0r0\\20211021150245_CT_ISRA_0.dcm")
test = Import_data("C:\\Users\\Werk\\OneDrive - UGent\\Burgie 4\\Sem 2\\VOP\\Coding\\microCT data X-Cube\\584_l0r0\\20211021150245_CT_ISRA_0.dcm")

#load and shuffle data
trainset = torch.utils.data.DataLoader(train,batch_size=10,shuffle=True)
testset = torch.utils.data.DataLoader(test,batch_size=10,shuffle=False)