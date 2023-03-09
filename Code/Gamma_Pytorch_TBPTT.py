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
'''=========================================================================='''
# Utilitiy functions
'''=========================================================================='''
'''=========================================================================='''


'''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''
# Function to generate the coeffficient of filter, which is used to generate
# X_k[n] from X[n]
'''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''
def g_k_filter_coeff(mu, k_list, filter_len):
    result = torch.zeros(filter_len, 1).type(torch.cuda.FloatTensor)
    
    for k in k_list:
        if k == 0:           # Case when k = 0, g_0[n] = delta[n], i.e. X_0[n] = X[n]
            result_ = torch.zeros_like(result)
            result_[0, 0] = 1.0
        else:    
            time_index = torch.arange(k, filter_len)
            nCk = torch.Tensor(comb(time_index-1, k-1))
            one_minus_mu_pow = torch.pow(1-mu, (time_index-k).type(torch.cuda.FloatTensor))
            result_ = nCk.type(torch.cuda.FloatTensor)*one_minus_mu_pow
            result_ = result_*torch.pow(mu, k.type(torch.cuda.FloatTensor))
            result_ = torch.cat((torch.zeros(int(k.numpy())).type(torch.cuda.FloatTensor), result_))
        result = torch.cat((result, result_.reshape(-1, 1)), dim = 1)
    result = result[:, 1:]    
    return result

'''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''
# Function to generate X_k[n] from X[n] 
'''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''
def generate_X_k_from_X(signal, mu = torch.tensor(1.0), K_list = torch.arange(5), filter_coeff_len = 50):

    signal_len = len(signal)
    signal_ = torch.unsqueeze(torch.unsqueeze(signal, 0), 0)
    result = torch.zeros_like(signal.reshape(-1, 1))
    gk_filter_coeff = g_k_filter_coeff(mu, K_list, filter_coeff_len)
    
    conv_func = nn.Conv1d(in_channels = 1, out_channels = 1, padding = (filter_coeff_len-1), kernel_size = filter_coeff_len, bias = False)

    conv_func.weight.requires_grad = False
    for k in torch.arange(len(K_list)):
        
        conv_filter = gk_filter_coeff[:, k].flip(0)
        conv_filter = torch.unsqueeze(torch.unsqueeze(conv_filter, 0), 0)
        conv_func.weight.data = conv_filter
        assert conv_func.weight.requires_grad == False
        
        result_ = conv_func(signal_)
        
        result_ = result_[0][0][:signal_len].reshape(-1, 1)
        result = torch.cat((result, result_), 1)
    result = result[:, 1:]  
    return result


'''-----------------------------------------------------------------------------'''
'''=============================Only for Gamma process==========================='''
'''============================================================================='''
# generate data for Gamma Process
'''============================================================================='''
'''-----------------------------------------------------------------------------'''


def generate_signal_given_mu(ip_signal, mu = 1.0, filter_order = 5, basis_weight = torch.ones(1, 6)):    
    X_k = torch.zeros(len(ip_signal), filter_order+1)
    X_k[:, 0] = ip_signal.reshape(-1)
    
    for i in torch.arange(1, len(ip_signal)):
       X_k[i, 1:] = mu*X_k[i-1, :-1] + (1 - mu)*X_k[i-1, 1:]
    result = torch.mm(X_k, basis_weight.transpose(dim0 = 0, dim1 = 1)).double()   + torch.tensor(np.random.normal(scale = .5, size = (len(ip_signal), 1)))
#    result = torch.mm(X_k, basis_weight.transpose(dim0 = 0, dim1 = 1)).double()
    return ip_signal.view(-1, 1), result.float()

ar_generator = 20
weight = torch.rand(1, ar_generator + 1)
weight[0] = 0.3
MCD_data_x, MCD_data_y = generate_signal_given_mu(torch.sin(torch.arange(0, 3000*np.pi, 0.1)),
                                              mu = .1, filter_order = ar_generator,
                                              basis_weight = weight)
'''============================================================================='''
# generate data for Gamma Process
'''============================================================================='''
'''=========================END!!!! Only for Gamma process==========================='''




'''=========================================================================='''
# Preparing the data
'''=========================================================================='''
ar_min = 1
ar_max = 15

step_pred_min = 1
step_pred_max = 20

np.random.seed(2121231)
intial_values = np.random.uniform()
MCD_data, time_step = generate_Mackey_data(x0 = intial_values, tau = 25, n = 20, beta = .5, gamma = .2, tmax = 2000000, t_step = 50)
MCD_data = np.array(MCD_data)

ar_list = np.arange(4)+1
DNN_R2 = np.zeros((4, 20))


for it in np.arange(3):
    print('------------->', it)
    ar = ar_list[it]
    # Data format:
    #         x[n-k] x[n-(k-1)] ... x[n-1] x[n]  
    #   time
    #    |
    #    V
    #    |
    #    V
    #    |
    #    V

    '''=========================================================================='''
    '''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''
    #    For Gamma Process
    '''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''    
    '''=========================================================================='''
    
#    MCD_data_AR_dat = generate_lag_k_steps_ahead_data(MCD_data_y.numpy(), ar_ord = ar, step_pred_min = 1, step_pred_max = 50) 
#    MCD_data_AR_dat = np.concatenate((MCD_data_x.numpy(), MCD_data_AR_dat), axis = 1)
#    MCD_data_AR_dat = MCD_data_AR_dat[10000:, :]


    '''=========================================================================='''
    '''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''
    #    For Macket-Glass Process
    '''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''    
    '''=========================================================================='''
    MCD_data_AR_dat = generate_lag_k_steps_ahead_data(MCD_data, ar_ord = ar, step_pred_min = 1, step_pred_max = 50)    
    MCD_data_AR_dat = MCD_data_AR_dat[10000:, :]
    
    '''=========================================================================='''
    # Final train and test data
    '''=========================================================================='''
    tr_X, tr_Y = MCD_data_AR_dat[:int(0.7*len(MCD_data_AR_dat)), 0], MCD_data_AR_dat[:int(0.7*len(MCD_data_AR_dat)), 1]
    te_X, te_Y = MCD_data_AR_dat[int(0.7*len(MCD_data_AR_dat)):, 0], MCD_data_AR_dat[int(0.7*len(MCD_data_AR_dat)):, 1]
    te_Y_full = MCD_data_AR_dat[int(0.7*len(MCD_data_AR_dat)):, 1:]
    tr_X, tr_Y = torch.from_numpy(tr_X).type(torch.cuda.FloatTensor).reshape(-1, 1), torch.from_numpy(tr_Y).type(torch.cuda.FloatTensor).reshape(-1, 1)
    te_X, te_Y = torch.from_numpy(te_X).reshape(-1, 1), torch.from_numpy(te_Y).reshape(-1, 1).reshape(-1, 1)
    
    '''=========================================================================='''
    # Defining the models
    '''=========================================================================='''
    
    # Non deep model; y[n] = a_0*X0[n] + a_1*X1[n] + a_2*X2[n] ... + a_k*Xk[n]
    # This is training in online mode i.e X_k[n] = (1-mu)X_k[n-1] + mu*X_(k-1)[n-1]
    
    #       X_k[n-1]          X_(k+1)[n-1] 
    #         |   \              |     \ 
    #   (1-mu)|    \mu       (1-mu)     \
    #         |     \            |       \
    #         |      \           |        \
    #         v       v          v         v
    #       X_k[n]     X_k[n]  X_(k+1)[n]
    
    class Gamma_mdl1(nn.Module):
        def __init__(self, in_feat, out_feat, order, block_size):
            super().__init__()    
            self.past_ip = torch.zeros(1, order+1).type(torch.cuda.FloatTensor)
            self.linear = nn.Linear(in_features = (order+1), out_features = out_feat)
            self.block_size = block_size
            self.mu = nn.Parameter(torch.tensor(1.0))
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, ip_signal):
            final_ip = self.past_ip.view(1, -1)       
            for j in range(self.block_size):                
                new_ip = self.mu*(final_ip[-1, :].roll(1)) + (1-self.mu)*final_ip[-1, :] 
                new_ip = new_ip.view(1, -1)
                new_ip[0, 0] = ip_signal[j, 0]
                final_ip = torch.cat((final_ip, new_ip))
    
            result =  self.linear(final_ip[1:, :])   
            self.past_ip = final_ip[1, :].view(1, -1).detach()
            return result
    
    class Gamma_mdl2(nn.Module):
        def __init__(self, in_feat, out_feat, order, block_size):
            super().__init__()    
            self.past_ip = torch.zeros(1, order+1).type(torch.cuda.FloatTensor)
            self.linear1 = nn.Linear(in_features = (order+1), out_features = 80)
            self.linear2 = nn.Linear(in_features = 80, out_features = 50)
            self.linear3 = nn.Linear(in_features = 50, out_features = out_feat)
            self.Relu = nn.ReLU()
            self.block_size = block_size
            self.sigmoid = nn.Sigmoid()
            self.mu = nn.Parameter(torch.tensor(1.0))
            
        def forward(self, ip_signal):
            final_ip = self.past_ip.view(1, -1)       
            for j in range(self.block_size):                
                new_ip = self.mu*(final_ip[-1, :].roll(1)) + (1-self.mu)*final_ip[-1, :] 
                new_ip = new_ip.view(1, -1)
                new_ip[0, 0] = ip_signal[j, 0]
                final_ip = torch.cat((final_ip, new_ip))
                
            result =  self.linear1(final_ip[1:, :])
            result =  self.Relu(result)
            result =  self.linear2(result)
            result =  self.Relu(result)
            result =  self.linear3(result)
            
            self.past_ip = final_ip[1, :].view(1, -1).detach()
            return result
        
    block_size = 1   
    
    net1 = Gamma_mdl2(in_feat = ar+1, out_feat = 1, order = ar, block_size = block_size).cuda()        
    optimizer = torch.optim.RMSprop(net1.parameters(), lr=0.004, weight_decay = 0.00001, momentum=0.8)
    loss_func = torch.nn.MSELoss()                      # this is for regression mean squared loss
    '''=========================================================================='''
    # Training the model
    '''=========================================================================='''
    net1.train()
    for i in np.arange(block_size, tr_X.shape[0]):        
        prediction = net1(tr_X[(i - block_size):i, :])                   # input x and predict based on x
#        loss = loss_func(prediction, tr_Y[(i - block_size):i, :])        # must be (1. nn output, 2. target)
        loss = loss_func(prediction[-1], tr_Y[i, :])        # must be (1. nn output, 2. target)
        
        optimizer.zero_grad()                           # clear gradients for next train
        loss.backward(retain_graph=True)                # backpropagation, compute gradients
        optimizer.step()                                # apply gradients  

        if i%2000 == 0:
#            print(net1.mu.item())
            optimizer.param_groups[0]['lr'] *= 0.95
            mu_new = 2.0*net1.sigmoid(net1.mu.data)
            res = generate_X_k_from_X(signal=te_X.type(torch.cuda.FloatTensor).reshape(-1),
                                      mu = net1.mu.data,
                                      K_list= torch.arange(ar+1),
                                      filter_coeff_len=70)
            val_pred = net1.linear3(net1.Relu(net1.linear2(net1.Relu(net1.linear1(res))))) 
            #val_pred = net1.linear(res)
            print(i, metrics.r2_score(te_Y, val_pred.cpu().detach().numpy()), net1.mu.item())

            
    '''=========================================================================='''
    # Generating k-steps ahead prediction and measuring the R2
    '''=========================================================================='''
    net1.eval()
    mu_new = 2.0*net1.sigmoid(net1.mu.data)
    res = generate_X_k_from_X(signal=te_X.type(torch.cuda.FloatTensor).reshape(-1),
                              mu = net1.mu.data,
                              K_list= torch.arange(ar+1, dtype = torch.int64),
                              filter_coeff_len=70)
#    Prediction_Final = net1.linear(res).cpu().detach()
    Prediction_Final = net1.linear3(net1.Relu(net1.linear2(net1.Relu(net1.linear1(res))))).cpu().detach()  
    
    Final_pred = te_X.cpu().detach()
    
    for i in range(step_pred_max):
        res = generate_X_k_from_X(signal=Final_pred.type(torch.cuda.FloatTensor).reshape(-1),
                              mu = net1.mu.data,
                              K_list= torch.arange(ar+1, dtype = torch.int64),
                              filter_coeff_len=50)
#        val_pred0 = net1.linear(res).cpu().detach()
        val_pred0 = net1.linear3(net1.Relu(net1.linear2(net1.Relu(net1.linear1(res))))).cpu().detach()  
        Prediction_Final = torch.cat((Prediction_Final, val_pred0), dim = 1)
        Final_pred = torch.roll(Final_pred, shifts = -1, dims = 1)
        Final_pred[:, -1] = val_pred0.reshape(-1)
    
    Prediction_Final = Prediction_Final[:, 1:]
    Prediction_Final = Prediction_Final.cpu().detach().numpy()
    DNN_R2_list = np.array([metrics.r2_score(te_Y_full[:, i], Prediction_Final[:, i]) for i in np.arange(0, step_pred_max)])
    DNN_R2[it, :] = DNN_R2_list
    
np.save('./Out_Data/Gamma_Linear_R2.npy', DNN_R2, allow_pickle=True, fix_imports=True)       


'''============================================================================================================================'''

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

import matplotlib
matplotlib.rcParams.update({'font.size': 22})
name_list = ['AR_Lin', 'AR_DNN', 'Laguerre_DNN', 'Laguerre_Lin', 'Gamma_Lin', 'Gamma_DNN']
title_list = ['AR Linear', 'AR DNN', 'Laguerre DNN', 'Laguerre Linear', 'Gamma Linear', 'Gamma DNN']
for name, file_title in zip(name_list, title_list):
    data_path = './Final_results/' + name + '_R2_noisy_tr_noisy_te' 
    data = np.load(data_path + '.npy')
    display_plot(data[:3], size = (16, 9), metric = 'R2', title = 'MG Attractor - ' + file_title + ' - noisy train, noisy test',
                 dst_name = data_path + '.png', save_flg = 1)
 