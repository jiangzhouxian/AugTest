
#pytorch libraries
import torch
from torchvision.models import resnet18,resnet50,wide_resnet50_2
from torch import nn
from torch.nn import init, Linear, ReLU, Softmax
import torch.nn.functional as F

rs18=resnet18(pretrained=True, progress = True) #这里采用resnet18模型
rs50=resnet50(pretrained=True, progress = True) #这里采用resnet18模型
wrs50=wide_resnet50_2(pretrained=True, progress = True) #这里采用resnet18模型

class r18(nn.Module):
    def __init__(self, model):
        super(r18,self).__init__()
        self.rn18 = model
        self.fl1 = nn.Linear(1000, 256)
        self.fl2 = nn.Linear(256,10)
        
    def forward(self, X):
        X = self.rn18(X)
        X = F.relu(self.fl1(X))
        X = F.dropout(X, p=0.25)
        X = self.fl2(X)
        return X
    
class r50(nn.Module):
    def __init__(self, model):
        super(r50,self).__init__()
        self.rn50 = model
        self.fl1 = nn.Linear(1000, 256)
        self.fl2 = nn.Linear(256,10)
        
    def forward(self, X):
        X = self.rn50(X)
        X = F.relu(self.fl1(X))
        X = F.dropout(X, p=0.25)
        X = self.fl2(X)
        return X
    
class wr50(nn.Module):
    def __init__(self, model):
        super(wr50,self).__init__()
        self.wrn50 = model
        self.fl1 = nn.Linear(1000, 256)
        self.fl2 = nn.Linear(256,10)
        
    def forward(self, X):
        X = self.wrn50(X)
        X = F.relu(self.fl1(X))
        X = F.dropout(X, p=0.25)
        X = self.fl2(X)
        return X