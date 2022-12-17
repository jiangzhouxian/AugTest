import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
import pandas as pd 
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ImageNet(Dataset):
    def __init__(self, annotations_file, img_dir , transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)[["Path","ClassId"]]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        label = self.img_labels.iloc[idx, 1]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
def float_transform(tensor):
    return tensor.float()
#图片转换规则
# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])  
 
unloader = transforms.ToPILImage()
#PIL转Tensor
# 输入图片地址, 返回tensor变量
def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
def PIL_to_tensor(image):
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

#Tensor转PIL
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

#直接展示tensor图像
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

#直接保存Tensor格式图片    
def save_image(tensor, **para):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    
    
#Numpy转化为Tensor
def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

#Tensor转Numpy
def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img

#展示Numpy图片
def show_from_cv(img, title=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

#展示Tensor格式图片   
def show_from_tensor(tensor, title=None):
    img = tensor.clone()
    img = tensor_to_np(img)
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)