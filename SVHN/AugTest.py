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



#step = args.step # 0.5
T = args.T #5
max_search = args.max_search #5
num_adv = 0
ave_diver = 0 #样本多样性分数

test_error1 = 0
test_error2 = 0
lambda_1 = args.weight_entropy #1
lambda_2 = args.weight_diversity #3

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
        geo = GeometricLoss()
        geo_ = Geometric()
        g = np.zeros(shape=(6,))#
        for iterate in range(T): 
            x = modelA(img)
            predictions = modelB(x)
            loss_1 = criterion(predictions, label)
            #print(loss_1)
            
            #Extract the feature vector of the last convolutional layer of VGG16, shape=（512，14，14）
            transform = Resize((224,224))
            resized_img = transform(img)
            x_fea = modelA(resized_img)
            fea = feature_Extract_net(x_fea)
            feat = inter_feature['features[28]'] 
            
            #Diversity loss and scores
            loss_2 = geo(feat)
            div = geo_(feat)
            #print(loss_2)
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
                    #对抗样本迁移性测试
                    pre_model_test1.eval()
                    pre_model_test2.eval()
                    
                    test1 = pre_model_test1(gen_img[bat].reshape(1,3,32,32))
                    _,test1_pre = torch.topk(test1, 1)
                    test_la1 = test1_pre[0][0]
                    
                    test2 = pre_model_test2(gen_img[bat].reshape(1,3,32,32))
                    _,test2_pre = torch.topk(test2, 1)
                    test_la2 = test2_pre[0][0]
                    
                    if test_la1 != orig_label[bat]:
                        test_error1 +=1
                    if test_la2 != orig_label[bat]:
                        test_error2 +=1              
                    #保存对抗样本
                    adv_img = gen_img[bat]
                    adv_label = pred_label_random[bat]
                    adv_img = adv_img.squeeze(0)
                    orig_ = orig_img[bat]
                    orig_ = orig_.squeeze(0)
                    adv_img_save = transforms.ToPILImage()(adv_img)
                    orig_img_save = transforms.ToPILImage()(orig_)
                    adv_img_save.save('{}adv_{}_{}_{}.jpg'.format(save_dir,str(num_adv),str(orig_label[1].numpy()),str(pred_label_random[bat][0].numpy())))
                    orig_img_save.save('{}orig_{}_{}.jpg'.format(orig_dir,str(num_adv),str(orig_label[1].numpy())))
            
            
            total_loss = lambda_1*loss_1 + lambda_2*loss_2
            total_loss.backward()
            
            #Calculating gradients
            grads = []
            for tt in modelA.parameters():
                #print(tt)
                grad = tt.grad
                grad = grad.numpy()
                grads.append(grad[0])
            grads = np.array(grads)
            grads = grads / np.mean(np.abs(grads))
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
            #print(theta,tx,sx,zx,beta,alpha)

            modelA = nn.Sequential(rotation_layer(theta),
                                   translate_layer(tx),
                                   shear_layer(sx),
                                   zoom_layer(zx),
                                   brightness_layer(beta),
                                   contrast_layer(alpha))

            gen_img = modelA(orig_img)
            modelB.eval()
            
            batch = gen_img.shape[0]
            for bat in range(batch):
                outputs = modelB(gen_img[bat].reshape(1,3,32,32))
                _,pred_label = torch.topk(outputs, 1)

                if pred_label[0]!= orig_label[bat]:
                    num_adv+=1
                    fea = feature_Extract_net(gen_img[bat].reshape(1,3,32,32))
                    feat = inter_feature['features[28]'] 
                    feat = feat.cpu()
                    div = geo_(feat)
                    diversity = div.clone()
                    diversity = diversity.detach().numpy()
                    #保存样本特征多样性分数
                    ave_diver += diversity
                    #对抗样本迁移性测试
                    pre_model_test1.eval()
                    pre_model_test2.eval()
                    
                    test1 = pre_model_test1(gen_img[bat].reshape(1,3,32,32))
                    _,test1_pre = torch.topk(test1, 1)
                    test_la1 = test1_pre[0][0]
                    
                    test2 = pre_model_test2(gen_img[bat].reshape(1,3,32,32))
                    _,test2_pre = torch.topk(test2, 1)
                    test_la2 = test2_pre[0][0]
                    
                    if test_la1 != orig_label[bat]:
                        test_error1 +=1
                    if test_la2 != orig_label[bat]:
                        test_error2 +=1              
                    #保存对抗样本
                    adv_img = gen_img[bat]
                    adv_label = pred_label
                    adv_img = adv_img.squeeze(0)
                    orig_ = orig_img[bat]
                    orig_ = orig_.squeeze(0)
                    adv_img_save = transforms.ToPILImage()(adv_img)
                    orig_img_save = transforms.ToPILImage()(orig_)
                    adv_img_save.save('{}adv_{}_{}_{}.jpg'.format(save_dir,str(num_adv),str(orig_label[1].numpy()),str(pred_label[0][0].numpy())))
                    orig_img_save.save('{}orig_{}_{}.jpg'.format(orig_dir,str(num_adv),str(orig_label[1].numpy())))
            img = gen_img
    '''
    gen_data_0 = np.array(gen_data_0)
    gen_data_1 = np.array(gen_data_1)
    #print(orig_img[0].shape,gen_data_0.shape)
    #print(orig_img[1].shape,gen_data_1.shape)
    geo_all = 0
    if gen_data_0.shape[0] != 0:
        geo_0 = Geodesic_distance(orig_img[0].reshape(-1,3,32,32),gen_data_0)
        geo_all += geo_0
    if gen_data_1.shape[0] != 0:
        geo_1 = Geodesic_distance(orig_img[1].reshape(-1,3,32,32),gen_data_1)
        geo_all += geo_1
   
    #print("geo distance of batchsize 2 are: ",geo_0,geo_1)
    ave_geo += geo_all
    '''
total_time = (datetime.datetime.now() - start_time).total_seconds()   
print('num_adv=',num_adv)
print('ave_diver = ',ave_diver/num_adv)
#print('ave_geo = ',ave_geo/50)
print('error_1,error_2=',test_error1,test_error2)
print('error_rate=',test_error1/num_adv,test_error2/num_adv)
print('Total time=',total_time)
print('ave_time=',total_time/num_adv)