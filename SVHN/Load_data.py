import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchvision.io import read_image
'''
class Gen_data(Dataset):
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

'''
class SVHNDataset(Dataset):
  def __init__(self,annotations_file,transform=None):
    
    self.img_path = pd.read_csv(annotations_file)[["Path"]]
    self.img_label = pd.read_csv(annotations_file)[["ClassId"]]
    if transform is not None:
      self.transform = transform
    else :
      transform=None
 
  def __getitem__(self,index):
    img = Image.open(self.img_path[index]).convert('RGB')
    if self.transform is not None:
      img = self.transform(img)
 
    # 原始SVHN中类别10为0
    labels = np.array(self.img_label[index],dtype=np.int)
    # 填充字符至5定长字符串
    labels = list(labels) + (5-len(labels))*[10]
    return img,torch.from_numpy(np.array(labels[:5]))
 
  def __len__(self):
    return len(self.img_path)