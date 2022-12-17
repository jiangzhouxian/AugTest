import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D

#计算样本之间的测地线距离
def Geodesic_distance(X_data):
    
    n_neighbours = 10
    n_components = 2
    Y = Isomap(n_neighbours,n_components).fit(X_data)
    return Y

    
    
    
    


   