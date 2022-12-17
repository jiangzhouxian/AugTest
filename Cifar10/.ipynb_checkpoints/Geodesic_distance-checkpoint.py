import numpy as np 
import pandas as pd 
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Pad, Compose,CenterCrop, ToPILImage, Normalize, ConvertImageDtype, Resize,Compose
from pandas import DataFrame
from PIL import Image
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from img_utils import *
from Aug_utils import *
from Geometric_loss import *
from metrics import *
    
def Geodesic_distance(seed,gen_data):

    train_sample = np.load("./npy_data/train_sample.npy")
    a = np.concatenate((seed,gen_data),axis=0)
    all_data = np.concatenate((a,train_sample),axis=0)
    all_data = all_data.reshape((all_data.shape[0],3072))
    isomap = Isomap(n_components=2,n_neighbors=5,path_method="auto")
    data_3d = isomap.fit_transform(X=all_data)
    geo_distance_metrix = isomap.dist_matrix_
    geo_distance = 0
    for i in range(len(gen_data)):
        geo_distance += geo_distance_metrix[0][i+1]
    ave_distance = geo_distance/len(gen_data)
    #print(ave_distance)
    return ave_distance
    