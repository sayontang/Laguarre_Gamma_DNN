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
import matplotlib.pyplot as plt
import gc
import matplotlib.pyplot as plt
from sklearn import metrics
import time
import torch.nn as nn
from Utilities import *

gc.disable()
gc.collect()


'''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''
# Function to generate X_k[n] from X[n] 
'''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''
def generate_X_k_from_X(signal, b = torch.tensor(0.0), K_max = 5):
    result = torch.zeros(signal.shape[0]+1, K_max)
    one_minus_b_sq = torch.sqrt(1-b**2)
    for i in range(1, result.shape[0]):
        result[i, 0] = b*result[i-1, 0] + one_minus_b_sq*signal[i-1, 0]
    
    for n in range(1, result.shape[0]):
        for k in range(1, K_max):
            result[n, k] = b*(result[n-1, k] - result[n, k-1]) + result[n-1, k-1]
    result = result[1:, :]        
    return result


'''=========================================================================='''
# Preparing the data
'''=========================================================================='''
ar_min = 1
ar_max = 15

step_pred_min = 1
step_pred_max = 15

np.random.seed(2121231)

'''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''
'''x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x'''
# generate data for Gamma Process
'''x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x'''
'''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''
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

'''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''
'''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''
intial_values = np.random.uniform()
MCD_data, time_step = generate_Mackey_data(x0 = intial_values, tau = 22, n = 20, beta = .5, gamma = .2, tmax = 150000, t_step = 1.0)
MCD_data = np.array(MCD_data)
ar_list = np.arange(4)+1
DNN_R2 = np.zeros((len(ar_list), step_pred_max+1))



for it in np.arange(1, len(ar_list)):
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
    
    MCD_data_AR_dat = generate_lag_k_steps_ahead_data(MCD_data_y.numpy(), ar_ord = ar, step_pred_min = 1, step_pred_max = 50) 
    MCD_data_AR_dat = np.concatenate((MCD_data_x.numpy(), MCD_data_AR_dat), axis = 1)
    MCD_data_AR_dat = MCD_data_AR_dat[10000:, :]

#    '''=========================================================================='''
#    '''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''
    #    For Mackey Glass Model
#    '''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'''    
#    '''=========================================================================='''
#    
#    MCD_data_AR_dat = generate_lag_k_steps_ahead_data(MCD_data, ar_ord = ar, step_pred_min = 1, step_pred_max = 50)    
#    MCD_data_AR_dat = MCD_data_AR_dat[50000:, :]
    
    '''=========================================================================='''
    # Final train and test data
    '''=========================================================================='''
    tr_X, tr_Y = MCD_data_AR_dat[:int(0.7*len(MCD_data_AR_dat)), 0], MCD_data_AR_dat[:int(0.7*len(MCD_data_AR_dat)), 1]
    te_X, te_Y = MCD_data_AR_dat[int(0.93*len(MCD_data_AR_dat)):, 0], MCD_data_AR_dat[int(0.93*len(MCD_data_AR_dat)):, 1]
    te_Y_full = MCD_data_AR_dat[int(0.93*len(MCD_data_AR_dat)):, 1:]
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
    
    class Laguarre_DNN_mdl(nn.Module):
        def __init__(self, in_feat, out_feat, order):
            super().__init__()    
            self.order = order
            self.past_ip = torch.zeros(1, self.order+1).type(torch.cuda.FloatTensor)            
            self.linear1 = nn.Linear(in_features = (order+1), out_features = 100)
            self.linear2 = nn.Linear(in_features = 100, out_features = 50)
            self.linear3 = nn.Linear(in_features = 50, out_features = out_feat)
            self.Relu = nn.ReLU()

            self.b = nn.Parameter(torch.tensor(0.1))
            
        def forward(self, ip_signal):
            new_ip = Variable(torch.zeros(1, self.order+1).type(torch.cuda.FloatTensor))             
            new_ip[0, 0] = self.b*self.past_ip[0, 0] + torch.sqrt(1 - self.b**2)*ip_signal
            
            for j in np.arange(1, self.order+1):
                new_ip[0, j] = self.b*(self.past_ip[0, j] - new_ip[0, j-1]) + self.past_ip[0, j-1]
                
            result =  self.linear1(new_ip)  
            result = self.Relu(result)
            result =  self.linear2(result)   
            result = self.Relu(result)
            result =  self.linear3(result)   
            self.past_ip = new_ip.detach_()
            return result
    
    
    class Laguarre_mdl(nn.Module):
        def __init__(self, in_feat, out_feat, order, block_size):
            super().__init__()    
            self.order = order
            self.past_ip = torch.zeros(1, self.order+1).type(torch.cuda.FloatTensor)
            self.linear = nn.Linear(in_features = (order+1), out_features = out_feat)
            self.b = nn.Parameter(torch.tensor(0.1))
            self.block_size = block_size
            
        def forward(self, ip_signal):
            final_ip = self.past_ip.view(1, -1) 
            
            for i in range(self.block_size):
                new_ip = Variable(torch.zeros(1, self.order+1).type(torch.cuda.FloatTensor))             
                new_ip[0, 0] = self.b*final_ip[-1, 0] + torch.sqrt(1 - self.b**2)*ip_signal[i]            
                for j in np.arange(1, self.order+1):
                    new_ip[0, j] = self.b*(self.past_ip[0, j] - new_ip[0, j-1]) + self.past_ip[0, j-1]
                final_ip = torch.cat((final_ip, new_ip))   
                
            result =  self.linear(final_ip[1:, :])   
            self.past_ip = final_ip[1, :].view(1, -1).detach()
            return result
                
        
    block_size = 5    
    net1 = Laguarre_mdl(in_feat = ar+1, out_feat = 1, order = ar, block_size = block_size).cuda()        
#    optimizer = torch.optim.Adam(net1.parameters(), lr=0.0003, weight_decay = 0.0001)
    optimizer = torch.optim.RMSprop(net1.parameters(), lr=0.0003, weight_decay = 0.00001, momentum = 0.9)
    loss_func = torch.nn.MSELoss()                      # this is for regression mean squared loss
    '''=========================================================================='''
    # Training the model
    '''=========================================================================='''
    net1.train()
    time_list2 = []
    for i in np.arange(block_size, 60000):        
        prediction = net1(tr_X[(i - block_size):i, :])                   # input x and predict based on x
        loss = loss_func(prediction[-1], tr_Y[i])        # must be (1. nn output, 2. target)
        optimizer.zero_grad()                           # clear gradients for next train
        loss.backward(retain_graph=True)                # backpropagation, compute gradients
        optimizer.step()                                # apply gradients  
        if i%500 == 0:
            print(i)
            time_list2.append(time.time())            
            res = generate_X_k_from_X(te_X,
                                      b = net1.b.data,
                                      K_max = ar+1)
            val_pred = net1.linear(res.cuda())
#            val_pred = net1.linear3(net1.linear2(net1.linear1(res.cuda())))
            print(i, metrics.r2_score(te_Y, val_pred.cpu().detach().numpy()), net1.b.item())
            optimizer.param_groups[0]['lr'] *= 0.9
            
    '''=========================================================================='''
    # Generating k-steps ahead prediction and measuring the R2
    '''=========================================================================='''
    net1.eval()
    res = generate_X_k_from_X(te_X,
                              b = net1.b.data,
                              K_max = ar+1)
#    Prediction_Final = net1.linear3(net1.Relu(net1.linear2(net1.Relu((net1.linear1(res.cuda()))))))
    Prediction_Final = net1.linear(res.cuda())

    for i in range(step_pred_max):
        res = generate_X_k_from_X(Prediction_Final[:, -1].view(-1, 1),
                                  b = net1.b.data,
                                  K_max = ar+1)
#        val_pred0 = net1.linear3(net1.Relu(net1.linear2(net1.Relu((net1.linear1(res.cuda()))))))
        val_pred0 = net1.linear(res.cuda())
        Prediction_Final = torch.cat((Prediction_Final.detach(), val_pred0), dim = 1)
    
    Prediction_Final = Prediction_Final.cpu().detach().numpy()
    DNN_R2_list = np.array([metrics.r2_score(te_Y_full[:, i], Prediction_Final[:, i]) for i in np.arange(Prediction_Final.shape[1])])
    DNN_R2[it, :] = DNN_R2_list
    
np.save('./Out_Data/Lagguare_DNN_R2_Full_SGD.npy', DNN_R2, allow_pickle=True, fix_imports=True)       


'''============================================================================================================================'''

def display_plot(data, size = (25, 12), metric = 'R2', title = 'Mackey-Glass Attractor', dst_name = 'MGD-R2.png', save_flg = 0):
    #Color scheme of the plot for different AR order        
    c = np.arange(data.shape[0])
    #Plotting the result
    plt.figure(figsize=size)    
    for i in c:
        plt.plot(np.arange(step_pred_min, (step_pred_max+2)), np.clip(data[i-c[0], :], 0, 1), label = '$Filter order = {i}$'.format(i = i+1))
    plt.xlabel('No. of steps ahead prediction')
    plt.ylabel(metric)
    plt.title(title)    
    plt.legend()
    if save_flg!=0:
        plt.savefig(dst_name)
    return   

data = np.load('./Out_Data/DNN_R2.npy')
display_plot(DNN_R2, size = (20, 14), metric = 'R2', title = 'Mackey-Glass Attractor - Lagaurre(Neural Network)', dst_name = './Out_Data/MGD-Lag_RMSprop.png', save_flg = 1)
 