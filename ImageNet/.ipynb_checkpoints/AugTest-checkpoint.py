#Prepare all our necessary libraries
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import argparse
#pytorch libraries
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Pad, Compose,CenterCrop, ToPILImage, Normalize, ConvertImageDtype, Resize,Compose

from torchvision.models import resnet50,wide_resnet50_2,vgg16

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
#from metrics import *
import datetime
# read the parameter
# argument parsing

parser = argparse.ArgumentParser(description='Main function for generation in ImageNet dataset')
parser.add_argument('model', help="model under tested", choices=[0, 1, 2], default =0,type = int)
parser.add_argument('decay', help ="decay factor of momentum", type=float, default = 1.0)

parser.add_argument('step_0', help="the step of gradient rise", type=float, default = 3)
parser.add_argument('step_1', help="the step of gradient rise", type=float, default = 0.01)
parser.add_argument('step_2', help="the step of gradient rise", type=float, default = 0.01)
parser.add_argument('step_3', help="the step of gradient rise", type=float, default = 0.04)
parser.add_argument('step_4', help="the step of gradient rise", type=float, default = 3.2)
parser.add_argument('step_5', help="the step of gradient rise", type=float, default = 0.04)

parser.add_argument('max_search', help="max search times", type=int, default = 5)
parser.add_argument('T', help="number of iterations of gradient rise", type=int, default = 10)
parser.add_argument('weight_entropy',help="weight of optimal entropy loss",type=float, default= 1.0)
parser.add_argument('weight_diversity',help="weight of optimal diversity loss",type=float, default= 1.0)
args = parser.parse_args()
args = parser.parse_args()

if args.model == 0:
    seed_file = "./seeds_resnet50.csv"
    pre_model = resnet50(pretrained=True)
    pre_model_test1 = wide_resnet50_2(pretrained=True)
    pre_model_test2 = vgg16(pretrained=True)
    subdir = 'ResNet50-ALL'
    #print("*****Test for ResNet50******** ")
elif args.model == 1:
    seed_file = "./seeds_wideresnet50.csv"
    pre_model = wide_resnet50_2(pretrained=True)
    pre_model_test1 = resnet50(pretrained=True)
    pre_model_test2 = vgg16(pretrained=True)
    subdir = 'WideResNet50-ALL'
    #print("*****Test for WideResNet50******** ")
elif args.model == 2:
    seed_file = "./seeds_vgg16.csv"
    pre_model = vgg16(pretrained=True)
    pre_model_test1 = resnet50(pretrained=True)
    pre_model_test2 = wide_resnet50_2(pretrained=True)
    subdir = 'VGG16-ALL'
    #print("*****Test for VGG16******** ")
feature_Extract_net =  vgg16(pretrained=True, progress = True)


img_dir = "./"
seed_data = ImageNet(img_dir = img_dir, annotations_file = seed_file,
                   transform = Compose([Resize((224,224)), ConvertImageDtype(torch.float32)]))
from torch.utils.data import DataLoader
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
    for search in range(max_search):
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
        #损失函数
        m = feature_Extract_net.features[28]
        inter_feature = {}
        def hook(m, input, output):
            inter_feature['features[28]'] = output
        m.register_forward_hook(hook)
        
        criterion = nn.CrossEntropyLoss()
        geo = GeometricLoss()
        g = np.zeros(shape=(6,))#
        for iterate in range(T): 
            x = modelA(img)
            x = x.cuda()
            predictions = modelB(x)
            #print(predictions)
            loss_1 = criterion(predictions, orig_label0)
            #Extract the feature vector of the last convolutional layer of VGG16, shape=（512，14，14）
            transform = Resize((224,224))
            resized_img = transform(img)
            x_fea = modelA(resized_img)
            fea = feature_Extract_net(x_fea)
            feat = inter_feature['features[28]'] 
            
            #Diversity loss and scores
            loss_2 = geo(feat)
            
            #print(loss_2)
            diversity = loss_2.clone()
            diversity = diversity.detach().numpy()
            total_loss = lambda_1*loss_1 + lambda_2*loss_2
            total_loss.backward()
            grads = []
            for tt in modelA.parameters():
                #print(tt)
                grad = tt.grad
                grad = grad.numpy()
                grads.append(grad[0])
            grads = np.array(grads)
            grads = grads / np.mean(np.abs(grads))
            #print(grads)
            #update g
            g = g * args.decay + grads
            #Gradient up update parameters
            theta += args.step_0* np.sign(g[0])
            tx += args.step_1* np.sign(g[1])
            sx += args.step_2* np.sign(g[2])
            zx += args.step_3* np.sign(g[3])
            beta += args.step_4* np.sign(g[4])
            alpha += args.step_5* np.sign(g[5])
            
            
            theta = np.clip(theta,-30,30)
            tx = np.clip(tx,-0.1,0.1)
            sx = np.clip(sx,-0.1,0.1)
            zx = np.clip(zx,0.8,1.2)
            beta = np.clip(beta,-32,32)
            alpha = np.clip(alpha,0.8,1.2)
            
            #更新图片
            modelA = nn.Sequential(rotation_layer(theta),
                                   translate_layer(tx),
                                   shear_layer(sx),
                                   zoom_layer(zx),
                                   brightness_layer(beta),
                                   contrast_layer(alpha))
            gen_img = modelA(orig_img)
            #复制一份
            gen_img_copy = gen_img.reshape(1,3,224,224)
            gen_img_copy = gen_img_copy.cuda()
            #再预测一次
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
                gen_data.append(gen_img_clone)
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
    if gen_data.shape[0] != 0:
        geo = Geodesic_distance(orig_img.reshape(-1,3,224,224),gen_data)
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