
#pytorch libraries
import torch
from torch import nn
from torch.nn import init, Linear, ReLU, Softmax
import torch.nn.functional as F


from torchvision.models import squeezenet1_0
squ = squeezenet1_0(pretrained=True, progress = True)

#adjust squeezenet1_0 to my dataset
class squeezenet(nn.Module):
    def __init__(self, pretrained_model):
        super(squeezenet,self).__init__()
        self.squ1 = pretrained_model
        self.fl1 = nn.Linear(1000, 256)
        self.fl2 = nn.Linear(256,43)
        
    def forward(self, X):
        X = self.squ1(X)
        X = F.relu(self.fl1(X))
        X = F.dropout(X, p=0.2)
        X = self.fl2(X)
        return X