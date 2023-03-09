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
            result_ = torch.cat((torch.zeros(k).type(torch.cuda.FloatTensor), result_))
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

'''=========================================================================='''
# Preparing the data
'''=========================================================================='''
ar_min = 1
ar_max = 15

step_pred_min = 1
step_pred_max = 20

np.random.seed(2121231)
intial_values = np.random.uniform()
MCD_data, time_step = generate_Mackey_data(x0 = intial_values, tau = 25, n = 20, beta = .5, gamma = .2, tmax = 3000000, t_step = 50)
MCD_data = np.array(MCD_data)

ar_list = np.arange(4)+1
DNN_R2 = np.zeros((4, 20))
DNN_R2_noisy = np.zeros((4, 20))

for it in np.arange(3)[:1]:
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
    
    MCD_data_AR_dat = generate_lag_k_steps_ahead_data(MCD_data, ar_ord = ar, step_pred_min = 1, step_pred_max = 50)
    MCD_data_AR_dat = MCD_data_AR_dat[8000:, :]
    
    '''=========================================================================='''
    # Final train and test data
    '''=========================================================================='''
    tr_X, tr_Y = MCD_data_AR_dat[:int(0.7*len(MCD_data_AR_dat)), 0], MCD_data_AR_dat[:int(0.7*len(MCD_data_AR_dat)), 1]
    te_X, te_Y = MCD_data_AR_dat[int(0.7*len(MCD_data_AR_dat)):, 0], MCD_data_AR_dat[int(0.7*len(MCD_data_AR_dat)):, 1]
    te_Y_full = MCD_data_AR_dat[int(0.7*len(MCD_data_AR_dat)):, 1:]
    tr_X_noisy = tr_X + np.random.normal(size = tr_X.shape, scale = 0.3)

    tr_X, tr_Y = torch.from_numpy(tr_X).type(torch.cuda.FloatTensor).reshape(-1, 1), torch.from_numpy(tr_Y).type(torch.cuda.FloatTensor).reshape(-1, 1)
    tr_X_noisy = torch.from_numpy(tr_X_noisy).type(torch.cuda.FloatTensor).reshape(-1, 1)
    te_X, te_Y = torch.from_numpy(te_X).reshape(-1, 1), torch.from_numpy(te_Y).reshape(-1, 1).reshape(-1, 1)
    te_X_noisy = te_X + torch.tensor(np.random.normal(size = te_X.shape, scale = 0.3))

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
        def __init__(self, in_feat, out_feat, order):
            super().__init__()    
            self.past_ip = torch.zeros(1, order+1).type(torch.cuda.FloatTensor)
            self.linear = nn.Linear(in_features = (order+1), out_features = out_feat)
            self.mu = nn.Parameter(torch.tensor(1.0))
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, ip_signal):
    #        mu2 = self.sigmoid(self.mu)       
    #        new_ip = (2.0*mu2*(self.past_ip.roll(1))) + ((1-2.0*mu2)*self.past_ip)               
            new_ip = self.mu*(self.past_ip.roll(1)) + (1-self.mu)*self.past_ip    
            new_ip[0, 0] = Variable(ip_signal)
            result =  self.linear(new_ip)   
            self.past_ip = new_ip.detach_()
            return result
    
    class Gamma_mdl2(nn.Module):
        def __init__(self, in_feat, out_feat, order):
            super().__init__()    
            self.past_ip = torch.zeros(1, order+1).type(torch.cuda.FloatTensor)
            self.linear1 = nn.Linear(in_features = (order+1), out_features = 80)
            self.linear2 = nn.Linear(in_features = 80, out_features = 50)
            self.linear3 = nn.Linear(in_features = 50, out_features = out_feat)
            self.Relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.mu = nn.Parameter(torch.tensor(1.0))
            
        def forward(self, ip_signal):
            new_ip = self.mu*(self.past_ip.roll(1)) + (1-self.mu)*self.past_ip    
            new_ip[0, 0] = Variable(ip_signal)
            result =  self.linear1(new_ip)
            result =  self.Relu(result)
            result =  self.linear2(result)
            result =  self.Relu(result)
            result =  self.linear3(result)
            self.past_ip = new_ip.detach_()
            return result
        
    net1 = Gamma_mdl2(in_feat = ar+1, out_feat = 1, order = ar).cuda()        
    optimizer = torch.optim.RMSprop(net1.parameters(), lr=0.001, weight_decay = 0.00001, momentum=0.8)
    loss_func = torch.nn.MSELoss()                      # this is for regression mean squared loss
    '''=========================================================================='''
    # Training the model
    '''=========================================================================='''
    net1.train()
#    time_list2 = []
    for i in np.arange(tr_X.shape[0]):        
        prediction = net1(tr_X_noisy[i, :])                   # input x and predict based on x
        loss = loss_func(prediction[0], tr_Y[i])        # must be (1. nn output, 2. target)
        optimizer.zero_grad()                           # clear gradients for next train
        loss.backward(retain_graph=True)                # backpropagation, compute gradients
        optimizer.step()                                # apply gradients  

        if i%2000 == 0:
#            time_list2.append(time.time())            
            print(i, net1.mu.data)
            mu_new = 2.0*net1.sigmoid(net1.mu.data)
            res = generate_X_k_from_X(signal=te_X_noisy.type(torch.cuda.FloatTensor).reshape(-1),
                                      mu = net1.mu.data,
                                      K_list= torch.arange(ar+1, dtype = torch.int64),
                                      filter_coeff_len=50)
            val_pred = net1.linear3(net1.Relu(net1.linear2(net1.Relu(net1.linear1(res))))) 
#            val_pred = net1.linear(res)
            print(i, metrics.r2_score(te_Y, val_pred.cpu().detach().numpy()), net1.mu.data)
            
    '''=========================================================================='''
    # Generating k-steps ahead prediction and measuring the R2
    '''=========================================================================='''
    net1.eval()
    mu_new = 2.0*net1.sigmoid(net1.mu.data)
    res = generate_X_k_from_X(signal=te_X.type(torch.cuda.FloatTensor).reshape(-1),
                              mu = net1.mu.data,
                              K_list= torch.arange(ar+1, dtype = torch.int64),
                              filter_coeff_len=50)
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

    '''=========================================================================='''
    # Generating k-steps ahead prediction and measuring the R2 - NOISY DATA
    '''=========================================================================='''
    net1.eval()
    mu_new = 2.0*net1.sigmoid(net1.mu.data)
    res = generate_X_k_from_X(signal=te_X_noisy.type(torch.cuda.FloatTensor).reshape(-1),
                              mu = net1.mu.data,
                              K_list= torch.arange(ar+1, dtype = torch.int64),
                              filter_coeff_len=50)
#    Prediction_Final = net1.linear(res).cpu().detach()
    Prediction_Final = net1.linear3(net1.Relu(net1.linear2(net1.Relu(net1.linear1(res))))).cpu().detach()  
    
    Final_pred = te_X_noisy.cpu().detach()
    
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
    DNN_R2_noisy[it, :] = DNN_R2_list
    
np.save('./Final_results/Gamma_DNN_R2_noisy_tr_clean_te.npy', DNN_R2, allow_pickle=True, fix_imports=True)       
np.save('./Final_results/Gamma_DNN_R2_noisy_tr_noisy_te.npy', DNN_R2_noisy, allow_pickle=True, fix_imports=True)       


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


#data = np.load('./Out_Data/DNN_R2.npy')
#display_plot(data, size = (20, 12), metric = 'R2', title = 'Mackey-Glass Attractor - AR(Neural Network)', dst_name = './Out_Data/MGD-AR_DNN.png', save_flg = 1)
 