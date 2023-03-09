# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 21:08:10 2019

@author: sayon
"""

import os
os.chdir('D:\workspace\pytorch\Lorenz_Mackey_DNN')
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from Utilities import *
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.special import comb
import gc
import matplotlib.pyplot as plt
from sklearn import metrics

gc.disable()
gc.collect()

'''=========================================================================='''
# #### Display performace metric plots
'''=========================================================================='''
def display_plot(data, size = (25, 12), metric = 'RMSE', title = 'Mackey-Glass Attractor', dst_name = 'MGD-RMSE.png', save_flg = 0):
    #Color scheme of the plot for different AR order        
    c = np.arange(14, 24)
    #Plotting the result
    plt.figure(figsize=size)    
    for i in c:
        plt.plot(np.arange(step_pred_min, (step_pred_max+1)), data[i-c[0], :], label = '$Filter order = {i}$'.format(i = i))
    plt.xlabel('No. of steps ahead prediction')
    plt.ylabel(metric)
    plt.title(title)    
    plt.legend()
    plt.autoscale(tight = True)
    if save_flg!=0:
        plt.savefig(dst_name)
    return   
'''=========================================================================='''
# Define Gamma filter coefficient: 
# g_k(n) = (n-1)Choose(k-1)*power(mu, k)*power(1-mu, n-k)*u(n-k) 
'''=========================================================================='''
def g_k(n, k, mu):
    return comb(n-1, k-1)*np.power(mu, k)*np.power(1-mu, n-k)*float(n>=k)

# Generate the X_k[n] variable for X[n] by convolving X[n]*G_k[n] and return the X_k[n] variable 
def generate_Gamma_filter_coeff(data, n = 50, K = 10, mu = 0.5):
    G_K_filter_matrix = np.zeros((n, 1))
    G_K_filter_matrix[0, 0] = 1.0
    
    for k in np.arange(1, K+1):
        vfunc = np.vectorize(g_k)
        G_k_filter = vfunc(n = np.arange(k, n), k = k, mu = mu)
        G_k_filter = np.append(np.zeros(k), G_k_filter)
        G_k_filter = G_k_filter.reshape((len(G_k_filter), 1))
        G_K_filter_matrix = np.concatenate((G_K_filter_matrix, G_k_filter), axis = 1)
        
    G_K_filter_matrix = G_K_filter_matrix/G_K_filter_matrix.sum(axis = 0)       #Normalizing the coefficient of G_k s.t they add upto 1.0

    X_K_matrix = data.copy()
    for k in np.arange(1, K+1):
        X_k = np.convolve(data.reshape(-1), G_K_filter_matrix[:, k], 'full')
        X_k = X_k[:len(data)]
        X_k = X_k.reshape(data.shape)
        X_K_matrix = np.concatenate((X_K_matrix, X_k), axis = 1)
    return X_K_matrix, G_K_filter_matrix

'''=========================================================================='''
# Define alpha_k using the following recursion: 
# alpha_k[n] = mu*alpha_k[n-1] + (1-mu)*alpha_{k-1}[n-1] + mu*[X_{k-1}[n-1] - X_k[n-1]]
'''=========================================================================='''
def generate_alpha_matrix(X_k_matrix, mu):
    alpha_k_matrix = np.zeros(X_k_matrix.shape) 
    nrow, ncol = X_k_matrix.shape
    
    X_diff_matrix = X_k_matrix[:(nrow-1), :(ncol-1)] - X_k_matrix[:(nrow-1), 1:]
    X_diff_matrix = mu*X_diff_matrix
    
    for r in np.arange(1, nrow):
        alpha_k_matrix[r, 1:] = mu*alpha_k_matrix[r-1, :(ncol-1)] + (1-mu)*alpha_k_matrix[r-1, 1:] + mu*X_diff_matrix[r-1, :]
    return alpha_k_matrix    

'''=========================================================================='''
#Generate predictions give the weights and the X_k variables
'''=========================================================================='''
def generate_output(X_k_matrix, weight_matrix):
    return np.matmul(X_k_matrix, weight_matrix)
        
'''=========================================================================='''
#Define the optimization routing with the updation rule for mu and the weights
'''=========================================================================='''
def optimize_Gamma_filter(tr_data, mu, weight, gk_filter_len = 55, filter_ord = 3, no_epochs = 500, mu_tstep = 0.1, w_tstep = 0.1):
    nrow, _ = tr_data.shape
    X_K_matrix, G_K_filter_matrix = generate_Gamma_filter_coeff(data = tr_data[:,0].reshape((nrow, 1)),
                                                                n = gk_filter_len, K = filter_ord, mu = mu)
    weight_result = weight.copy()

    mu_result = np.copy(mu)
    for epochs in range(no_epochs):
        X_K_matrix = X_K_matrix[gk_filter_len + 10:, :]
        
        alpha_matrix = generate_alpha_matrix(X_K_matrix, mu_result)
        y_pred = generate_output(X_K_matrix, weight_result)
        
        error = tr_data[gk_filter_len + 10:, 1].reshape(y_pred.shape) - y_pred
        
        d_W = np.multiply(X_K_matrix, error).mean(axis = 0)
        d_mu = np.multiply(error, np.matmul(alpha_matrix, weight_result)).mean(axis = 0)
        
        weight_result = weight_result + w_tstep*d_W.reshape(weight_result.shape)
        mu_result = mu_result + mu_tstep*d_mu
        if epochs%20 == 0:
            print(epochs, (error**2).mean(), mu_result)
        X_K_matrix, G_K_filter_matrix = generate_Gamma_filter_coeff(data = tr_data[:,0].reshape((nrow, 1)),
                                                                    n = gk_filter_len, K = filter_ord, mu = mu_result)
    return weight_result, mu_result    

def optimize_Gamma_filter_v2(tr_data, mu, weight, gk_filter_len = 55, filter_ord = 3, no_epochs = 500, mu_tstep = 0.1, w_tstep = 0.1):
    nrow, _ = tr_data.shape
    partition = np.arange(0, nrow, 10000)
    weight_result = weight.copy()

    mu_result = np.copy(mu)
    for epochs in range(no_epochs):
        for p in np.arange(len(partition)-1):
            ind1, ind2 = partition[p], partition[p+1]
            X_K_matrix, G_K_filter_matrix = generate_Gamma_filter_coeff(data = tr_data[ind1:ind2, 0].reshape((ind2 - ind1, 1)),
                                                                n = gk_filter_len, K = filter_ord, mu = mu)
            X_K_matrix = X_K_matrix[gk_filter_len + 10:, :]
            
            alpha_matrix = generate_alpha_matrix(X_K_matrix, mu_result)
            y_pred = generate_output(X_K_matrix, weight_result)
            y_act = tr_data[ind1:ind2:, 1]
            y_act = y_act[gk_filter_len + 10:].reshape(y_pred.shape)
            
            error = y_act - y_pred
            
            d_W = np.multiply(X_K_matrix, error).mean(axis = 0)
            d_mu = np.multiply(error, np.matmul(alpha_matrix, weight_result)).mean(axis = 0)
            
            weight_result = weight_result + w_tstep*d_W.reshape(weight_result.shape)
            mu_result = mu_result + mu_tstep*d_mu
            if epochs%20 == 0:
                print(epochs, (error**2).mean(), mu_result)

    return weight_result, mu_result  

'''=========================================================================='''
#Generate the K-steps ahead prediction for the test data using the estimated weights and mu
'''=========================================================================='''
def generate_prediction(te_data, weights, no_of_steps = 50, filter_ord = 5, mu = 0.41, gk_filter_len = 55):
    nrow, _ = te_data.shape
    predictions = np.zeros((len(te_data), 1))
    te_dat = te_data[:,0]
    for steps in np.arange(no_of_steps):
        X_K_matrix, G_K_filter_matrix = generate_Gamma_filter_coeff(data = te_dat.reshape((nrow, 1)),
                                                                    n = gk_filter_len, K = filter_ord, mu = mu)
        pred_temp = generate_output(X_K_matrix, weights)
        predictions = np.concatenate((predictions, pred_temp), axis = 1)
        te_dat = pred_temp.copy()  
    predictions = predictions[:, 1:]    
    return predictions

    
#def plot_autocorrelation():
#    dt_list = [.1, .5, 1, 2, 5, 10]
#    intial_values = np.random.uniform(0, 3, 10)
#    init = 0
#    Corr_matrix = np.zeros((len(dt_list), 20))
#    i = 0
#    for dt in dt_list:
#        print(dt)
#        MCD_data, time_step = generate_Mackey_data(x0 = intial_values[init], tau = 22, n = 20, beta = .5, gamma = .2, tmax = 100000, t_step = dt)
#        lag_data = generate_lag_k_steps_ahead_data(MCD_data, ar_ord = 0, step_pred_max = 20)
#        display_Mackey_data(lag_data, time_step, size = (10, 10))
#        lag_data = lag_data[2000:, ]
#        cov_matrix = np.corrcoef(lag_data.T)
#        Corr_matrix[i, :] = cov_matrix[:, i]
#        i = i+1
        
        
'''=========================================================================='''
#Generate Mackey-Glass data
'''=========================================================================='''
filter_order_list = np.arange(14, 24)
epoch_list = [300, 300, 300, 200, 200, 100, 100, 100, 100, 100]
step_pred_max = 400
np.random.seed(2121231)
intial_values = np.random.uniform(0, 3, 10)

MCD_data, time_step = generate_Mackey_data(x0 = intial_values[0], tau = 22, n = 20, beta = .5, gamma = .2, tmax = 250000, t_step = 2)
MCD_data_k_lag_data = generate_lag_k_steps_ahead_data(MCD_data, ar_ord = 1, step_pred_max = 400)
MCD_data_k_lag_data = MCD_data_k_lag_data[30000:, :]

Gamma_R2_matrix = np.zeros((len(filter_order_list), 400))
Gamma_RMSE_matrix = np.zeros((len(filter_order_list), 400))

for filt_ord in filter_order_list[-5::1]:
    print('---->', filt_ord)
    weight_init = np.random.uniform(size = (filt_ord+1, 1))
    weight_init = weight_init/weight_init.sum()
    
    tr_end_index = int(0.7*len(MCD_data_k_lag_data))
    tr_data, te_data = MCD_data_k_lag_data[:tr_end_index, :2],  MCD_data_k_lag_data[tr_end_index:, :]
    
    weight_pred1, mu_pred = optimize_Gamma_filter(tr_data, mu = 1.0, weight = weight_init, gk_filter_len = 50,
                                                 filter_ord = filt_ord, no_epochs = epoch_list[filt_ord-filter_order_list[0]], mu_tstep = 0.1, w_tstep=0.12)
        
    te_pred = generate_prediction(MCD_data_k_lag_data, weight_pred1,
                                         no_of_steps = 400, filter_ord = filt_ord, mu = mu_pred, gk_filter_len = 50)
    te_pred = te_pred[tr_end_index:, :]
    
    Gamma_R2_list = np.array([metrics.r2_score(te_data[:, i+1], te_pred[:, i]) for i in range(step_pred_max)]) 
    Gamma_RMSE_list = np.array([calculate_RMSE(te_data[:, i+1], te_pred[:, i]) for i in range(step_pred_max)]) 

    Gamma_R2_matrix[filt_ord - filter_order_list[0], :] = Gamma_R2_list
    Gamma_RMSE_matrix[filt_ord - filter_order_list[0], :] = Gamma_RMSE_list
   
display_plot(np.clip(Gamma_R2_matrix, 0, 1), size = (40, 18), metric = 'R2', title = 'Mackey-Glass Gamma - dt = 2', dst_name = '.\Out_Data\Gamma_MC_R2_dt2_K400.png', save_flg = 1)
display_plot(np.clip(Linear_R2_matrix, 0, 1), size = (40, 18), metric = 'R2', title = 'Mackey-Glass LINEAR - dt = 2', dst_name = '.\Out_Data\Linear_MC_R2_dt2_K400.png', save_flg = 1)
