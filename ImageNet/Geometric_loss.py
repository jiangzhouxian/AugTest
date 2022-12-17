import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch import nn
import numpy as np

det_offset = 1e-6
#删除全0行
def delete_0_row(input_tensor):
    #x_data = torch.tensor([[1, 2], [0, 0], [1, 1]])
    x = input_tensor.clone()
    x = x.detach().numpy()
    flag_0_list = list()
    #print('原tensor： {}'.format(x_data))
    for i in range(0, input_tensor.size(0)):
        if np.all(np.array(x[i, :]) == 0):
            continue
        else:
            flag_0_list.append(i)
    indices = torch.LongTensor(np.array(flag_0_list))
    return torch.index_select(input_tensor, 0, indices)

#计算几何多样性损失
class GeometricLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,input_tensor):
        V = input_tensor  #（2，512，14，14）
        batch_size = input_tensor.shape[0] #batch size
        feature_num = input_tensor.shape[1] #特征数量 
        all_log_det_list = list()
        m,n=0,0
        for i in range(batch_size):
            V_x = V[i]
         
            for j in range(feature_num):
                min_ = V_x[j].min().item()
                max_ = V_x[j].max().item()

                if(min_ == max_): #跳过全0特征
                    continue

                pre_matrix = delete_0_row(V_x[j])#删除全0行

                pre_matrix = (pre_matrix-min_)/(max_ - min_)#归一化

                matrix = torch.matmul(pre_matrix,pre_matrix.permute(1,0))#计算内积
                all_log_det = torch.logdet(matrix+det_offset) #计算每个特征的多样性分数 
                
                #print(all_log_det)
                if(torch.isnan(all_log_det)==False and torch.isinf(all_log_det)==False):
                    all_log_det = torch.abs(all_log_det)#绝对值
                    all_log_det_list.append(all_log_det)
        #平均多样性分数
        ave_logdet = torch.mean(torch.stack(all_log_det_list))
        #print(ave_logdet)
        #all_log_det = all_log_det/feature_num #平均多样性分数
        
        return ave_logdet

#样本多样性
class Geometric(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,input_tensor):
        V = input_tensor  #（2，512，14，14）
        batch_size = input_tensor.shape[0] #batch size
        all_det_list = list()
        m,n=0,0
        for i in range(batch_size):
            V_x = V[i]
            pre_matrix = torch.flatten(V_x)#展平
            min_ = pre_matrix.min().item()
            max_ = pre_matrix.max().item()
            pre_matrix = (pre_matrix-min_)/(max_ - min_)#归一化
            all_det = torch.square(torch.norm(pre_matrix))
            all_det_list.append(all_det)
        ave_logdet = torch.mean(torch.stack(all_det_list))
        
        return ave_logdet
