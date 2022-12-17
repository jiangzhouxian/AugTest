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


#save generated data to csv files
for i in range(3):
    if i == 0:
        filepath = './generated_inputs/ResNet50-ALL/'
    elif i == 1:
        filepath = './generated_inputs/WideResNet50-ALL/'
    elif i == 2:
        filepath = './generated_inputs/SqueezeNet-ALL/'
    
    gen_data = []

    for file in os.listdir(filepath):
        if file.split(".")[-1] == 'jpg':
            a = file.split('_')
            data = [a[2],os.path.join(filepath,file)]
            gen_data.append(data)

    df = pd.DataFrame(gen_data, columns= ["ClassId","Path"])
    if i == 0:
        df.to_csv("./csv_data/gen_rs50.csv")
        gen_file = "./csv_data/gen_rs50.csv"
        data_3 = pd.read_csv(gen_file,usecols=["Path","ClassId"])
    elif i == 1:
        df.to_csv("./csv_data/gen_wrs50.csv")
        gen_file = "./csv_data/gen_wrs50.csv"
        data_3 = pd.read_csv(gen_file,usecols=["Path","ClassId"])
    elif i == 2:
        df.to_csv("./csv_data/gen_squ.csv")
        gen_file = "./csv_data/gen_squ.csv"
        data_3 = pd.read_csv(gen_file,usecols=["Path","ClassId"])
    


    train_file = "./Models/Dataset/train_sample.csv"
    data_1 = pd.read_csv(train_file,usecols=["Path","ClassId"])
    data_1["Path"]=['./Models/Dataset/%s' % i for i in data_1["Path"]]
    #print(data_1)
    seed_file = "./Models/Dataset/seed50.csv"
    data_2 = pd.read_csv(seed_file,usecols=["Path","ClassId"])
    data_2["Path"]=['./Models/Dataset/%s' % i for i in data_2["Path"]]

    frames = [data_1,data_2,data_3]
    result = pd.concat(frames)
    data=DataFrame(result,columns=["Path","ClassId"])
    data.to_csv("./csv_data/all_data.csv")

    img_path = ''
    all_data = GTSRB(img_dir = img_path, annotations_file = "./csv_data/all_data.csv",
                    transform = Compose([Resize((30,30)), ConvertImageDtype(torch.float32)]))

    print(len(all_data))

    all_dataloader = DataLoader(all_data, batch_size=len(all_data), shuffle=False)
    all_features, all_labels = next(iter(all_dataloader))
    batch = all_features.shape[0]
    all_features = all_features.view((batch,2700))
    isomap = Isomap(n_components=2,n_neighbors=5,path_method="auto")
    data_3d = isomap.fit_transform(X=all_features)
    geo_distance_metrix = isomap.dist_matrix_
    print(np.mean(geo_distance_metrix))