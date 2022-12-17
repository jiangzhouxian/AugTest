#pytorch libraries
import torch
from torchvision.models import vgg16
from torch import nn
from torch.nn import init, Linear, ReLU, Softmax
import torch.nn.functional as F

#downloading vgg16 pretrained on ImageNet 
VGGNet = vgg16(pretrained=True, progress = True)

#adjust vgg16 to my dataset
class vgg(nn.Module):
    def __init__(self, pretrained_model):
        super(vgg,self).__init__()
        self.vgg16 = pretrained_model
        self.fl1 = nn.Linear(1000, 256)
        self.fl2 = nn.Linear(256,43)
        
    def forward(self, X):
        X = self.vgg16(X)
        X = F.relu(self.fl1(X))
        X = F.dropout(X, p=0.25)
        X = self.fl2(X)
        return X