
#pytorch libraries
import torch
from torchvision.models import resnet50
from torch import nn
from torch.nn import init, Linear, ReLU, Softmax
import torch.nn.functional as F

#downloading resent50 pretrained on ImageNet 
resn50 = resnet50(pretrained=True, progress = True)

#adjust resnet50 to my dataset
class r50(nn.Module):
    def __init__(self, pretrained_model):
        super(r50,self).__init__()
        self.rn50 = pretrained_model
        self.fl1 = nn.Linear(1000, 256)
        self.fl2 = nn.Linear(256,43)
        
    def forward(self, X):
        X = self.rn50(X)
        X = F.relu(self.fl1(X))
        X = F.dropout(X, p=0.25)
        X = self.fl2(X)
        return X

net =r50(pretrained_model=resn50)

#print(net.named_children())


# register forward hook
# m = net.rn50.layer4[2].conv3
m = net.rn50.layer4[2].relu
inter_feature = {}
def hook(m, input, output):
    # inter_feature['rn50.layer4[2].conv3'] = output
    inter_feature['rn50.layer4[2].relu'] = output
m.register_forward_hook(hook)


# forward
#x = torch.rand(size=(2, 3, 30, 30)).cuda()
#net = net.cuda()
#out = net(x)

# feat = inter_feature['rn50.layer4[2].conv3']
#feat = inter_feature['rn50.layer4[2].relu']
#print(feat.shape)
#print(feat)
