# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 20:29:59 2019

@author: sayon
"""

import os
os.chdir('D:\workspace\pytorch\Lorenz_Mackey_DNN')
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from scipy.special import comb
import gc
import matplotlib.pyplot as plt
from sklearn import metrics
import time
import torch.nn as nn
from Utilities import *

gc.disable()
gc.collect()

'''=========================================================================='''
# Preparing the data
'''=========================================================================='''


ar = 2

ar_min = 1
ar_max = 15

step_pred_min = 1
step_pred_max = 20

DNN_R2 = np.zeros((3, 20))
ar_list = np.arange(1, 4)

np.random.seed(2121231)
intial_values = np.random.uniform()

# Generate Gamma process signal#

def generate_signal_given_mu(ip_signal, mu = 1.0, filter_order = 4, basis_weight = torch.ones(1, 6)):    
    X_k = torch.zeros(len(ip_signal), filter_order+1)
    X_k[:, 0] = ip_signal.reshape(-1)
    
    for i in torch.arange(1, len(ip_signal)):
       X_k[i, 1:] = mu*X_k[i-1, :-1] + (1 - mu)*X_k[i-1, 1:]
    result = torch.mm(X_k, basis_weight.transpose(dim0 = 0, dim1 = 1))   
    return ip_signal, result

# Generate the time series signal
MCD_data, time_step = generate_Mackey_data(x0 = intial_values, tau = 22, n = 20, beta = .5, gamma = .2, tmax = 100000, t_step = 1.0)

# Generate the Gamma time series signal
weight = torch.rand(1, ar+1)
MCD_data_x, MCD_data_y = generate_signal_given_mu(torch.sin(torch.arange(0, 3000*np.pi, 0.1)),
                                                  mu = 0.5, filter_order = ar,
                                                  basis_weight = weight)

MCD_data_x, MCD_data_y = MCD_data_x[15000:].view(-1, 1), MCD_data_y[15000:, :]
MCD_data_AR_dat = torch.cat((MCD_data_x, MCD_data_y), dim = 1).numpy()

for it in np.arange(3):
    print('------------>', it)
    ar = ar_list[it]
    
    '''=========================================================================='''
    '''=========================================================================='''
    # GAMMA PROCESS data
    '''=========================================================================='''
    '''=========================================================================='''
    MCD_data_AR_dat = generate_lag_k_steps_ahead_data(MCD_data_x.numpy(), ar_ord = ar, step_pred_min = 1, step_pred_max = 20)
    MCD_data_AR_dat_Y = generate_lag_k_steps_ahead_data(MCD_data_y.numpy(), ar_ord = ar, step_pred_min = 1, step_pred_max = 20)
    
    
    '''=========================================================================='''
    # Final train and test data
    '''=========================================================================='''
    tr_X, tr_Y = MCD_data_AR_dat[:int(0.7*len(MCD_data_AR_dat)), :ar], MCD_data_y.numpy()[:int(0.7*len(MCD_data_AR_dat_Y)), 0].reshape(-1, 1)
    te_X, te_Y = MCD_data_AR_dat[int(0.7*len(MCD_data_AR_dat)):, :ar], MCD_data_y.numpy()[int(0.7*len(MCD_data_AR_dat_Y)):, 0].reshape(-1, 1)
    te_Y_full = MCD_data_AR_dat_Y[int(0.7*len(MCD_data_AR_dat)):]
    
    tr_X, tr_Y = torch.from_numpy(tr_X).type(torch.cuda.FloatTensor), torch.from_numpy(tr_Y).type(torch.cuda.FloatTensor).reshape(-1, 1)
    te_X, te_Y = torch.from_numpy(te_X), torch.from_numpy(te_Y).reshape(-1, 1)

    
    '''=========================================================================='''
    '''=========================================================================='''
    # MACKEY-GLASS data
    '''=========================================================================='''
    '''=========================================================================='''
#    MCD_data_AR_dat = generate_lag_k_steps_ahead_data(MCD_data, ar_ord = ar, step_pred_min = 1, step_pred_max = 20)
#    MCD_data_AR_dat = MCD_data_AR_dat[20000:, :]
#    
#    '''=========================================================================='''
#    # Final train and test data
#    '''=========================================================================='''
#    tr_X, tr_Y = MCD_data_AR_dat[:int(0.7*len(MCD_data_AR_dat)), :ar], MCD_data_AR_dat[:int(0.7*len(MCD_data_AR_dat)), ar]
#    te_X, te_Y = MCD_data_AR_dat[int(0.7*len(MCD_data_AR_dat)):, :ar], MCD_data_AR_dat[int(0.7*len(MCD_data_AR_dat)):, ar]
#    te_Y_full = MCD_data_AR_dat[int(0.7*len(MCD_data_AR_dat)):, ar:]
#    tr_X, tr_Y = torch.from_numpy(tr_X).type(torch.cuda.FloatTensor), torch.from_numpy(tr_Y).type(torch.cuda.FloatTensor).reshape(-1, 1)
#    te_X, te_Y = torch.from_numpy(te_X), torch.from_numpy(te_Y).reshape(-1, 1)
    
    '''=========================================================================='''
    # Defining the models
    '''=========================================================================='''
    class Linear_mdl1(nn.Module):
        def __init__(self, in_feat, out_feat):
            super().__init__()            
            self.linear = nn.Linear(in_features =  in_feat, out_features = out_feat)
                 
        def forward(self, ip_signal):
            result =  self.linear(ip_signal)   
            return result
    
    class Linear_mdl2(nn.Module):
        def __init__(self, in_feat, out_feat):
            super().__init__()            
            self.linear1 = nn.Linear(in_features = in_feat, out_features = 30)
            self.linear2 = nn.Linear(in_features = 30, out_features = 15)
            self.linear3 = nn.Linear(in_features = 15, out_features = out_feat)
            self.Relu = nn.ReLU()
                 
        def forward(self, ip_signal):
            result = self.linear1(ip_signal) 
            result = self.Relu(result)
            result = self.linear2(result)  
            result = self.Relu(result)
            result = self.linear3(result)
            return result
    
    net1 = Linear_mdl1(in_feat = ar, out_feat = 1).cuda()        
    optimizer = torch.optim.RMSprop(net1.parameters(), lr=0.00001, weight_decay = 0.0001)
    loss_func = torch.nn.MSELoss()  
    
    '''=========================================================================='''
    # Training the model
    '''=========================================================================='''
    time_list = []
    for i in np.arange(tr_X.shape[0]):
        prediction = net1(tr_X[i, :])             # input x and predict based on x
        loss = loss_func(prediction, tr_Y[i])     # must be (1. nn output, 2. target)
        optimizer.zero_grad()                     # clear gradients for next train
        loss.backward()                           # backpropagation, compute gradients
        optimizer.step()                          # apply gradients  
        if i%5000 == 0:
            optimizer.param_groups[0]['lr'] *= 0.7
            Prediction_Final =  net1(te_X.type(torch.cuda.FloatTensor)).cpu().detach()
            print(metrics.r2_score(te_Y_full[:, 0], Prediction_Final))
#            print(i)
    
    '''=========================================================================='''
    # Generating k-steps ahead prediction and measuring the R2
    '''=========================================================================='''
    Prediction_Final =  net1(te_X.type(torch.cuda.FloatTensor)).cpu().detach()
    
    Final_pred = te_X.cpu().detach()
    
    for i in range(step_pred_max):
        val_pred0 = net1(Final_pred.type(torch.cuda.FloatTensor)).cpu().detach()
        Prediction_Final = torch.cat((Prediction_Final, val_pred0), dim = 1)
        Final_pred = torch.roll(Final_pred, shifts = -1, dims = 1)
        Final_pred[:, -1] = val_pred0.reshape(-1)
    
    Prediction_Final = Prediction_Final[:, 1:]
    Prediction_Final = Prediction_Final.cpu().detach().numpy()
    DNN_R2_list = np.array([metrics.r2_score(te_Y_full[:, i], Prediction_Final[:, i]) for i in np.arange(0, step_pred_max)])
    DNN_R2[it, :] = DNN_R2_list
    

def display_plot(data, size = (25, 12), metric = 'R2', title = 'Mackey-Glass Attractor', dst_name = 'MGD-RMSE.png', save_flg = 0):
    #Color scheme of the plot for different AR order        
    c = np.arange(data.shape[0])
    #Plotting the result
    plt.figure(figsize=size)    
    for i in c:
        plt.plot(np.arange(step_pred_min, (step_pred_max+1)), np.clip(data[i-c[0], :], 0, 1), label = '$Filter order = {i}$'.format(i = i+1))
    plt.xlabel('No. of steps ahead prediction')
    plt.ylabel(metric)
    plt.title(title)    
    plt.legend()
    if save_flg!=0:
        plt.savefig(dst_name)
    return

display_plot(DNN_R2, size = (20, 12), metric = 'R2', title = 'Gamma Process - Linear AR', dst_name = './Results/G-AR-DNN.png', save_flg = 0)
       
np.save('./Result/Linear_R2.npy', DNN_R2, allow_pickle=True, fix_imports=True)        