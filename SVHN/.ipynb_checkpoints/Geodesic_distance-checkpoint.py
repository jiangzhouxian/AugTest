import numpy as np 
import pandas as pd 
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Pad, Compose,CenterCrop, ToPILImage, Normalize, ConvertImageDtype, Resize,Compose
from pandas import DataFrame
from torchvision.io import read_image
from torch.utils.data import DataLoader
from img_utils import *
from Aug_utils import *
from Geometric_loss import *
from metrics import *
from Load_data import *


#save generated data to csv files
for i in range(3):
    if i == 0:
        filepath = './generated_inputs/Geometric_ResNet50/'
    elif i == 1:
        filepath = './generated_inputs/Geometric_WideResNet50/'
    elif i == 2:
        filepath = './generated_inputs/Geometric_SqueezeNet/'
    
    gen_data = []

    for file in os.listdir(filepath):
        if file.split(".")[-1] == 'jpg':
            a = file.split('_')
            data = [a[2],os.path.join(filepath,file)]
            gen_data.append(data)

    df = pd.DataFrame(gen_data, columns= ["ClassId","Path"])
    img_path = ''
    transform = transforms.Compose([transforms.ToTensor()])
    if i == 0:
        df.to_csv("./csv_data/gen_rs50.csv")
        gen_file = "./csv_data/gen_rs50.csv"
        data_3 = pd.read_csv(gen_file,usecols=["Path","ClassId"])
        gen_data = Gen_data(img_dir = img_path, annotations_file = gen_file,
                    transform = transform)
    elif i == 1:
        df.to_csv("./csv_data/gen_wrs50.csv")
        gen_file = "./csv_data/gen_wrs50.csv"
        data_3 = pd.read_csv(gen_file,usecols=["Path","ClassId"])
        gen_data = Gen_data(img_dir = img_path, annotations_file = gen_file,
                    transform = transform)
    elif i == 2:
        df.to_csv("./csv_data/gen_squ.csv")
        gen_file = "./csv_data/gen_squ.csv"
        data_3 = pd.read_csv(gen_file,usecols=["Path","ClassId"])
        gen_data = Gen_data(img_dir = img_path, annotations_file = gen_file,
                    transform = transform)

    gen_dataloader = torch.utils.data.DataLoader(gen_data, batch_size=len(gen_data),shuffle=False)
    gen_features, gen_labels = next(iter(gen_dataloader))


    train_dataset = SVHN(root='./Dataset', split='train',download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset),shuffle=False)
    train_features, train_labels = next(iter(train_dataloader))

    test_dataset = SVHN(root='./Dataset', split='test',download=True, transform=transform)
    seed_data = []
    for i in range(0,1000,20):
        seed_data.append(test_dataset[i])
    seed_dataloader = torch.utils.data.DataLoader(seed_data, batch_size=len(seed_data),shuffle=False)
    seed_features, seed_labels = next(iter(seed_dataloader))

    all_data = np.vstack((gen_features,train_features))
    all_data = np.vstack((all_data,seed_features))
    print(len(all_data))
    batch = all_data.shape[0]
    all_data = all_data.view((batch,3072))
    isomap = Isomap(n_components=2,n_neighbors=5,path_method="auto")
    data_3d = isomap.fit_transform(X=all_data)
    geo_distance_metrix = isomap.dist_matrix_
    print(np.mean(geo_distance_metrix))