#Prepare all our necessary libraries
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
from PIL import Image
import argparse
#pytorch libraries
import torch
from torchvision.transforms import Compose,ConvertImageDtype,Resize,RandomResizedCrop,Normalize,ToPILImage,ToTensor

from torchvision.models import resnet50,wide_resnet50_2,vgg16
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from img_utils import *
from Aug_utils import *
from Geometric_loss import *
from Geodesic_distance import *

import datetime

parser = argparse.ArgumentParser(description='Main function for generation in ImageNet dataset')
parser.add_argument('model', help="model under tested", choices=[0, 1, 2], default =0,type = int)
parser.add_argument('decay', help ="decay factor of momentum", type=float, default = 1.0)

parser.add_argument('max_search', help="max search times", type=int, default = 5)
parser.add_argument('T', help="number of iterations of gradient rise", type=int, default = 10)
args = parser.parse_args()
args = parser.parse_args()

if args.model == 0:
    seed_file = "./seeds_resnet50.csv"
    pre_model = resnet50(pretrained=True)
    pre_model_test1 = wide_resnet50_2(pretrained=True)
    pre_model_test2 = vgg16(pretrained=True)
    subdir = 'ResNet50-ALL'
    print("*****Test for ResNet50******** ")
elif args.model == 1:
    seed_file = "./seeds_wideresnet50.csv"
    pre_model = wide_resnet50_2(pretrained=True)
    pre_model_test1 = resnet50(pretrained=True)
    pre_model_test2 = vgg16(pretrained=True)
    subdir = 'WideResNet50-ALL'
    print("*****Test for WideResNet50******** ")
elif args.model == 2:
    seed_file = "./seeds_vgg16.csv"
    pre_model = vgg16(pretrained=True)
    pre_model_test1 = resnet50(pretrained=True)
    pre_model_test2 = wide_resnet50_2(pretrained=True)
    subdir = 'VGG16-ALL'
    print("*****Test for VGG16******** ")
feature_Extract_net =  vgg16(pretrained=True, progress = True)

img_dir = "./"
seed_data = ImageNet(img_dir = img_dir, annotations_file = seed_file,
                   transform = Compose([Resize((224,224)),ConvertImageDtype(torch.float32)]))
seed_dataloader = DataLoader(seed_data, batch_size=1, shuffle=False)
test_features, test_labels = next(iter(seed_dataloader))

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

    
    #step = args.step # 0.5
T = args.T #5
max_search = args.max_search #5
num_adv = 0
find_time = []
ave_diver = 0 #样本多样性分数

test_error1 = 0
test_error2 = 0
ave_geo = 0
lambda_1 = args.weight_entropy #1
lambda_2 = args.weight_diversity #3

for i, data in enumerate(seed_dataloader, 0):
    img, label = data
    start_time = datetime.datetime.now()
    orig_img = img
    test_img = img
    gen_data = []
    #原始图片和标签
    pre_model.cuda()
    pre_model.eval() 
    pre_model_test1.cuda()
    pre_model_test2.cuda()
    pre_model_test1.eval()
    pre_model_test2.eval()
    with torch.no_grad():
        #img = img.reshape(1,3,224,224)
        test_img = test_img.cuda()
        outputs0 = pre_model(test_img)
        outputs1 = pre_model_test1(test_img)
        outputs2 = pre_model_test2(test_img)
        _,pred0 = torch.topk(outputs0, 1)
        _,pred1 = torch.topk(outputs1, 1)
        _,pred2 = torch.topk(outputs2, 1)
        
    orig_label0,orig_label1,orig_label2 = pred0[0],pred1[0],pred2[0] #对于pre_model的原始标签
    search_times = 0
    for search in range(max_search*T):
        #搜索次数初始化
        theta = np.random.randint(-30,30)
        tx = np.random.uniform(-0.1,0.1)
        sx = np.random.uniform(-0.1,0.1)
        zx = np.random.uniform(0.8,1.2)
        beta = np.random.uniform(-32,32)
        alpha = np.random.uniform(0.8,1.2)
        #print(theta,tx,sx,zx,beta,alpha)
        #初始化模型
        modelA = nn.Sequential(rotation_layer(theta),
                               translate_layer(tx),
                               shear_layer(sx),
                               zoom_layer(zx),
                               brightness_layer(beta),
                               contrast_layer(alpha))
        modelB = pre_model
        modelB.cuda()
       
           
        outputs_pre0 = modelB(gen_img_copy)
        _,pred_label_0 = torch.topk(outputs_pre0, 1)
        #print(orig_label,pred_label_1[0])
        if pred_label_0[0][0]!= orig_label0[0]:
            #print('Is adv!')
            num_adv+=1
            find_time.append((datetime.datetime.now() - start_time).total_seconds())
            start_time = datetime.datetime.now()
            ave_diver += diversity
            gen_img_clone = gen_img.clone()
            gen_img_clone = gen_img_clone.detach().numpy()
            gen_data.append(gen_img_clone[0])
            #对抗样本迁移性测试
            pre_model_test1.cuda()
            pre_model_test2.cuda()
            pre_model_test1.eval()
            pre_model_test2.eval()
                
            test1 = pre_model_test1(gen_img_copy)
            _,test1_pre = torch.topk(test1, 1)
            test_la1 = test1_pre[0][0]
                
            test2 = pre_model_test2(gen_img_copy)
            _,test2_pre = torch.topk(test2, 1)
            test_la2 = test2_pre[0][0]
                
            if test_la1 != orig_label1[0]:
                test_error1 +=1
            if test_la2 != orig_label2[0]:
                test_error2 +=1              
                #保存对抗样本
            adv_img = gen_img
            adv_label = orig_label0
            adv_img = adv_img.squeeze(0)
            orig_ = orig_img
            orig_ = orig_.squeeze(0)
            adv_img_save = transforms.ToPILImage()(adv_img)
            orig_img_save = transforms.ToPILImage()(orig_)
        adv_img_save.save('{}adv_{}_{}_{}.jpg'.format(save_dir,str(num_adv),str(orig_label0[0].cpu().numpy()),str(pred_label_0[0][0].cpu().numpy())))
                orig_img_save.save('{}orig_{}_{}.jpg'.format(orig_dir,str(num_adv),str(orig_label0[0].cpu().numpy())))

            img = gen_img
    gen_data = np.array(gen_data)
    #print(gen_data.shape)
    if (gen_data.shape[0] != 0):
        geo = Geodesic_distance(orig_img.reshape(-1,3,224,224),gen_data)
        #print(geo)
        ave_geo += geo
find_time = np.array(find_time)       
print('num_adv=',num_adv)
print('ave_diver = ',ave_diver/num_adv)
print('ave_geo = ',ave_geo/50)
print('error_1,error_2=',test_error1,test_error2)
print('error_rate=',test_error1/num_adv,test_error2/num_adv)
print('Total time=',np.sum(find_time))
print('ave_time=',np.mean(find_time))

#print(num_adv,test_error1,test_error1/num_adv,test_error2,test_error2/num_adv,np.sum(ave_search_time)/num_adv,duration,duration/num_adv)