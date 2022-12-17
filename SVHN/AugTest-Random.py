#Prepare all our necessary libraries
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
#pytorch libraries
import torch
from torch.utils.data import Dataset
from torchvision.datasets import SVHN
from torchvision import datasets
from torchvision.transforms import ToTensor, Pad, Compose,CenterCrop, ToPILImage, Normalize, ConvertImageDtype, Resize,Compose

from Models.resnet50 import *
from Models.wide_resnet50 import *
from Models.squeezenet import *
from torchvision.models import resnet50,wide_resnet50_2,squeezenet1_0,vgg16

from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import init, Linear, ReLU, Softmax
from torch.nn.init import xavier_uniform_
from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from img_utils import *
from Aug_utils import *
from Geometric_loss import *
from Geodesic_distance import *
import datetime
import argparse

parser = argparse.ArgumentParser(description='Main function for generation in SVHN dataset')
parser.add_argument('model', help="model under tested", choices=[0, 1, 2], default =0,type = int)
parser.add_argument('max_search', help="max search times", type=int, default = 5)

args = parser.parse_args()

# Load seed inputs
transform = transforms.Compose(
    [transforms.ToTensor()])

test_dataset = SVHN(root='./Dataset', split='test',download=True, transform=transform)
seed_data = []
for i in range(0,1000,20):
    seed_data.append(test_dataset[i])
seed_dataloader = torch.utils.data.DataLoader(seed_data, batch_size=2,shuffle=False)
test_features, test_labels = next(iter(seed_dataloader))

# Load models under tested
if args.model == 0:
    subdir = 'Geometric_ResNet50'
    pre_model = r50(resn50)
    pre_model.load_state_dict(torch.load('./Models/svhn_resnet50.pth'))
    pre_model_test1 = wr50(wresn50)
    pre_model_test1.load_state_dict(torch.load('./Models/svhn_wide_resnet50.pth'))
    pre_model_test2 = squeezenet(squ)
    pre_model_test2.load_state_dict(torch.load('./Models/svhn_squeezenet.pth'))
    print("*****Test for ResNet50********")
    
elif args.model == 1:
    subdir = 'Geometric_WideResNet50'
    pre_model_test1 = r50(resn50)
    pre_model_test1.load_state_dict(torch.load('./Models/svhn_resnet50.pth'))
    pre_model = wr50(wresn50)
    pre_model.load_state_dict(torch.load('./Models/svhn_wide_resnet50.pth'))
    pre_model_test2 = squeezenet(squ)
    pre_model_test2.load_state_dict(torch.load('./Models/svhn_squeezenet.pth'))
    print("*****Test for WideResNet50********")
elif args.model == 2:
    subdir = 'Geometric_SqueezeNet'
    pre_model_test1 = r50(resn50)
    pre_model_test1.load_state_dict(torch.load('./Models/svhn_resnet50.pth'))
    pre_model_test2 = wr50(wresn50)
    pre_model_test2.load_state_dict(torch.load('./Models/svhn_wide_resnet50.pth'))
    pre_model = squeezenet(squ)
    pre_model.load_state_dict(torch.load('./Models/svhn_squeezenet.pth'))
    print("*****Test for SqueezeNet********")

feature_Extract_net =  vgg16(pretrained=True, progress = True)
#print(feature_Extract_net)
'''
#create path of images saving
save_dir = './generated_inputs/' + subdir + '/'
orig_dir = './generated_inputs/' + subdir + '/seeds/'
if os.path.exists(save_dir):
    for i in os.listdir(save_dir):
        path_file = os.path.join(save_dir, i)
        if os.path.isfile(path_file):
            os.remove(path_file)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
if os.path.exists(orig_dir):
    for i in os.listdir(orig_dir):
        path_file = os.path.join(orig_dir, i)
        if os.path.isfile(path_file):
            os.remove(path_file)

if not os.path.exists(orig_dir):
    os.makedirs(orig_dir)


'''
#step = args.step # 0.5
max_search = args.max_search #5
num_adv = 0
ave_diver = 0 #样本多样性分数

for i, data in enumerate(seed_dataloader, 0):
    img, label = data
    start_time = datetime.datetime.now()
    #origal images and label
    orig_img = img
    orig_label = label

    #initialization parameters
    
    for search in range(max_search):
        
        theta = np.random.randint(-30,30)
        tx = np.random.uniform(-0.1,0.1)
        sx = np.random.uniform(-0.1,0.1)
        zx = np.random.uniform(0.8,1.2)
        beta = np.random.uniform(-32,32)
        alpha = np.random.uniform(0.8,1.2)
        #print(theta,tx,sx,zx,beta,alpha)
        #initialization models
        modelA = nn.Sequential(rotation_layer(theta),
                               translate_layer(tx),
                               shear_layer(sx),
                               zoom_layer(zx),
                               brightness_layer(beta),
                               contrast_layer(alpha))
        modelB = pre_model
        #feature_net = feature_Extract_net
        m = feature_Extract_net.features[28]
        inter_feature = {}
        def hook(m, input, output):
            inter_feature['features[28]'] = output
    
        m.register_forward_hook(hook)
        #entropy loss
        criterion = nn.CrossEntropyLoss()
        #geo = GeometricLoss()
        geo_ = Geometric()
        #g = np.zeros(shape=(6,))#
        
        x = modelA(img)
        predictions = modelB(x)
        
        transform = Resize((224,224))
        resized_img = transform(img)
        x_fea = modelA(resized_img)
        fea = feature_Extract_net(x_fea)
        feat = inter_feature['features[28]'] 
            
        #Diversity loss and scores
        div = geo_(feat)
        diversity = div.clone()
        diversity = diversity.detach().numpy()
        _,pred_label_random = torch.topk(predictions, 1)
        gen_img = x.clone()
        #初始化就已经是对抗样本
        batch = pred_label_random.shape[0]
        for bat in range(batch):
            if pred_label_random[bat]!= orig_label[bat]:
                #print('Is adv!')
                num_adv+=1
                #保存样本特征多样性分数
                ave_diver += diversity
            
total_time = (datetime.datetime.now() - start_time).total_seconds()   
print('num_adv=',num_adv)
print('ave_diver = ',ave_diver/num_adv)
print('Total time=',total_time)
print('ave_time=',total_time/num_adv)