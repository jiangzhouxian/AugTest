import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
import torch
from ops.para2matrix import *

#Rotation
class rotation_layer(nn.Module):
    def __init__(self,q):
        super().__init__()
        self.theta = nn.Parameter(torch.Tensor([q * np.pi/180]))
    def forward(self, X):      
        M = theta2matrix(self.theta)
        batchSize = X.shape[0]
        M = M.repeat(batchSize,1,1)
        grid = F.affine_grid(M , torch.Size((batchSize,3,32,32)))
        output = F.grid_sample(X, grid)
        return output
    
#Translate
class translate_layer(nn.Module):
    def __init__(self,x):
        super().__init__()
        self.tx = nn.Parameter(x * torch.ones(1,))
        #self.ty = nn.Parameter(0.1 * torch.ones(1,))
        
    def forward(self, X):    
        # 构造一个Translate仿射矩阵
        M = tx2Matrix(self.tx)
        #M = torch.tensor([[1, 0, self.tx], [0,1,self.ty]])
        #print('M',M)
        batchSize = X.shape[0]
        #print('batchSize',batchSize)
        M = M.repeat(batchSize,1,1)
        grid = F.affine_grid(M , torch.Size((batchSize,3,32,32)))
        output = F.grid_sample(X, grid)
        return output
    
#Shear
class shear_layer(nn.Module):
    def __init__(self,x):
        super().__init__()
        self.sx = nn.Parameter(x * torch.ones(1,))
        #self.sy = nn.Parameter(0.1 * torch.ones(1,))
        
    def forward(self, X):
        # 构造一个Translate仿射矩阵
        M = sx2Matrix(self.sx)
        batchSize = X.shape[0]
        M = M.repeat(batchSize,1,1)
        grid = F.affine_grid(M , torch.Size((batchSize,3,32,32)))
        output = F.grid_sample(X, grid)
        return output
#Zoom    
class zoom_layer(nn.Module):
    def __init__(self,x):
        super().__init__()
        self.zx = nn.Parameter(x * torch.ones(1,))
        #self.zy = nn.Parameter(0.8 * torch.ones(1,))
        
    def forward(self, X):  
        # 构造一个Translate仿射矩阵
        M = zx2Matrix(self.zx)
        batchSize = X.shape[0]
        M = M.repeat(batchSize,1,1)
        grid = F.affine_grid(M , torch.Size((batchSize,3,32,32)))
        output = F.grid_sample(X, grid)
        return output

# brightness
class brightness_layer(nn.Module):
    def __init__(self,b):
        super().__init__()
        self.beta = nn.Parameter(b * torch.ones(1,))
        
    def forward(self, X):
        batchSize = X.shape[0]
        #M = self.beta
        #M = M.repeat(batchSize,3,30,30)
        output = self.beta+X*255
        output[output>255]=255
        output[output<0]=0
        output = output/255
        return output

# contrast
class contrast_layer(nn.Module):
    def __init__(self,a):
        super().__init__()
        self.alpha = nn.Parameter(a * torch.ones(1,))
        
    def forward(self, X):
        batchSize = X.shape[0]
        #print('batchSize',batchSize)
        X = X*255
        output = self.alpha*X
        output[output>255]=255
        output[output<0]=0
        output = output/255
        return output