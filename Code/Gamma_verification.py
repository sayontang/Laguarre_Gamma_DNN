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

def generate_signal_given_mu(ip_signal, mu = 1.0, filter_order = 5, basis_weight = torch.ones(1, 6)):    
    X_k = torch.zeros(len(ip_signal), filter_order+1)
    X_k[:, 0] = ip_signal.reshape(-1)
    
    for i in torch.arange(1, len(ip_signal)):
       X_k[i, 1:] = mu*X_k[i-1, :-1] + (1 - mu)*X_k[i-1, 1:]
    result = torch.mm(X_k, basis_weight.transpose(dim0 = 0, dim1 = 1))   
    return ip_signal, result

'''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''
# Display the final plots
'''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''
    
def display_plot(data, size = (25, 12), metric = 'R2', title = 'Mackey-Glass Attractor', dst_name = 'MGD-RMSE.png', save_flg = 0):
    #Color scheme of the plot for different AR order        
    c = np.arange(data.shape[0])
    #Plotting the result
    plt.figure(figsize=size)  
    x_ticks = np.arange(data.shape[1])
    plt.plot(x_ticks, data[0, :], 'k--', label = 'True $\mu = {i}$'.format(i = data[0, 0]))
    for i in c[1:]:
        plt.plot(x_ticks, data[i, :], label = 'initial $\mu = {i}$'.format(i = data[i, 0]))
    plt.xlabel('No. of updates (x300)')
    plt.ylabel(metric)
    plt.title(title)    
    plt.legend()
    if save_flg!=0:
        plt.savefig(dst_name)
    return   

'''=========================================================================='''
# Preparing the data
'''=========================================================================='''
ar_min = 1
ar_max = 15

step_pred_min = 1
step_pred_max = 20

ar_list = np.arange(6)+1
true_mu_list = [1.25, .95, 1.25, 1.57, 1.97]


ar = 1
for mu_t in true_mu_list:

    Final_R2 = np.ones((1, 440))
    Final_mu = np.ones((1, 440))

    weight = torch.rand(1, ar+1)
    MCD_data_x, MCD_data_y = generate_signal_given_mu(torch.sin(torch.arange(0, 2000*np.pi, 0.1)),
                                                      mu = mu_t, filter_order = ar,
                                                      basis_weight = weight)
    MCD_data_x, MCD_data_y = MCD_data_x[5000:].view(-1, 1), MCD_data_y[5000:, :]
    MCD_data_AR_dat = torch.cat((MCD_data_x, MCD_data_y), dim = 1).numpy()
    r2_0 = metrics.r2_score(MCD_data_y, MCD_data_x)                                    #initial R2 between input and the targe
        
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
    mu_init_list = [0.01, 1.0, 1.099]
    for mu_ in mu_init_list:
        class Gamma_mdl1(nn.Module):
            def __init__(self, in_feat, out_feat, order):
                super().__init__()    
                self.past_ip = torch.zeros(1, order+1).type(torch.cuda.FloatTensor)
                self.linear = nn.Linear(in_features = (order+1), out_features = out_feat)
                self.linear.weight.requires_grad = False
                self.linear.weight.data = weight
                self.mu = nn.Parameter(torch.tensor(mu_))
                
            def forward(self, ip_signal):
                new_ip = self.mu*(self.past_ip.roll(1)) + (1-self.mu)*self.past_ip    
                new_ip[0, 0] = Variable(ip_signal)
                result =  self.linear(new_ip)   
                self.past_ip = new_ip.detach_()
                return result
        
        r2_list = []
        mu_list = []
            
        net1 = Gamma_mdl1(in_feat = ar+1, out_feat = 1, order = ar).cuda()        
        optimizer = torch.optim.RMSprop(net1.parameters(), lr=0.02)
        loss_func = torch.nn.MSELoss()                      # this is for regression mean squared loss
        '''=========================================================================='''
        # Training the model
        '''=========================================================================='''
        net1.train()
        for i in np.arange(tr_X.shape[0]):  
            if i%300 == 0:      
                optimizer.param_groups[0]['lr'] *= 0.9
                res = generate_X_k_from_X(signal=te_X.type(torch.cuda.FloatTensor).reshape(-1),
                                          mu = net1.mu.data,
                                          K_list= torch.arange(ar+1, dtype = torch.int32),
                                          filter_coeff_len=50)
                val_pred = net1.linear(res).cpu().detach()
                r2_list.append(metrics.r2_score(te_Y, val_pred.cpu().detach().numpy()))
                mu_list.append(net1.mu.item())
                print(i, mu_list[-1])
            prediction = net1(tr_X[i, :])                   # input x and predict based on x
            loss = loss_func(prediction[0], tr_Y[i])        # must be (1. nn output, 2. target)
            optimizer.zero_grad()                           # clear gradients for next train
            loss.backward(retain_graph=True)                # backpropagation, compute gradients
            optimizer.step()                                # apply gradients
        
#        r2_list = np.array(r2_list).reshape(1, -1)
#        Final_R2 = np.concatenate((Final_R2, r2_list), axis = 0)    
    
        mu_list = np.array(mu_list).reshape(1, -1)
        Final_mu = np.concatenate((Final_mu, mu_list), axis = 0) 
    
    Final_mu[0, :] = Final_mu[0, :]*mu_t
    display_plot(Final_mu, size = (20, 10), metric = '$\mu$',
                 title = 'Mackey-Glass Attractor-$\mu$ convergance', dst_name = 'mu_converg_'+str(mu_t)+'.png', save_flg = 1)

'''============================================================================================================================'''



data = np.load('./Out_Data/DNN_R2.npy')
display_plot(data, size = (20, 12), metric = 'R2', title = 'Mackey-Glass Attractor - AR(Neural Network)', dst_name = './Out_Data/MGD-AR_DNN.png', save_flg = 1)
 