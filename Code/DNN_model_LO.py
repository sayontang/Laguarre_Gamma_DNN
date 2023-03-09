# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 23:10:08 2019

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
from Utilities import *
import gc
import matplotlib.pyplot as plt
from sklearn import metrics

gc.disable()
gc.collect()

step_pred_max = 50
step_pred_min = 1
'''=========================================================================='''
# #### Display performace metric plots
'''=========================================================================='''
def display_plot(data, size = (25, 12), metric = 'RMSE', title = 'Mackey-Glass Attractor', dst_name = 'MGD-RMSE.png', save_flg = 0):
    #Color scheme of the plot for different AR order        
    c = np.linspace(0, 1, (ar_order_max - ar_order_min + 1))
    #Plotting the result
    plt.figure(figsize=size)    
    for i in range(len(c)):
        plt.plot(np.arange(step_pred_min, (step_pred_max+1)), data[i, :], label = '$AR order = {i}$'.format(i = i + ar_order_min))
    plt.xlabel('No. of steps ahead prediction')
    plt.ylabel(metric)
    plt.title(title)    
    plt.legend()
    plt.autoscale(tight = True)
    if save_flg!=0:
        plt.savefig(dst_name)
    return   
'''=========================================================================='''

BATCH_SIZE = 80000
EPOCH = 70

ar_min = 1
ar_max = 15
step_pred_max = 50

intial_values = np.random.uniform(0, 3, 10)
init = 0

Lorenz_data, time_step = generate_Lorenz_data(sigma = 10, beta = 2.667, rho = 28, u0 = 1.9, v0 = 1.3, w0 = 1.7, tmax = 10000, n = 300000)
Lorenz_data = Lorenz_data.T
display_Lorenz_data(Lorenz_data[150000:, :])

DNN_R2_matrix = np.zeros((ar_max - ar_min + 1, step_pred_max))
DNN_RMSE_matrix = np.zeros((ar_max - ar_min + 1, step_pred_max))
Linear_R2_matrix = np.zeros((ar_max - ar_min + 1, step_pred_max))
Linear_RMSE_matrix = np.zeros((ar_max - ar_min + 1, step_pred_max))

for ar in np.arange(1, 16):
    print(ar)
    Lor_data_AR_dat = generate_lag_k_steps_ahead_data(Lorenz_data, ar_ord = ar, step_pred_min = 1, step_pred_max = 50)
    Lor_AR_dat = Lor_data_AR_dat[100000:, :]
    
    '''========================================================================================================'''
    '''============================Linear model for 50 steps ahead prediction=================================='''
    '''========================================================================================================'''
    tr_X, tr_Y = Lor_AR_dat[:int(0.7*len(Lor_AR_dat)), :3*ar], Lor_AR_dat[:int(0.7*len(Lor_AR_dat)), 3*ar:(3*(ar+1))]
    te_X, te_Y = Lor_AR_dat[int(0.7*len(Lor_AR_dat)):, :3*ar], Lor_AR_dat[int(0.7*len(Lor_AR_dat)):, 3*ar:(3*(ar+1))]
    te_Y_full = Lor_AR_dat[int(0.7*len(Lor_AR_dat)):, 3*ar:]
    
    reg_mdl = linear_model.Ridge(alpha = 10.0).fit(tr_X, tr_Y)
        
    pred = reg_mdl.predict(te_X)
    R2 = metrics.r2_score(te_Y, pred)
    RMSE = calculate_RMSE(te_Y, pred)
    
    Prediction_Final =  reg_mdl.predict(te_X).reshape((len(te_X), 3))
     
    te_temp = te_X.copy()
    for i in range(50):
        val_pred0 = reg_mdl.predict(te_X).reshape((len(te_X), 3))
        Prediction_Final = np.concatenate((Prediction_Final, val_pred0), axis = 1)
        te_X = np.roll(te_X, -3, axis = 1)
        te_X[:, (3*(ar-1)):(3*ar)] = val_pred0
    
    Prediction_Final = Prediction_Final[:, 3:]
        
    Linear_R2_list = np.array([metrics.r2_score(te_Y_full[:, i:(i+3)], Prediction_Final[:, i:(i+3)]) for i in np.arange(0, step_pred_max*3, 3)]) 
    Linear_RMSE_list = np.array([calculate_RMSE(te_Y_full[:, i:(i+3)], Prediction_Final[:, i:(i+3)]) for i in np.arange(0, step_pred_max*3, 3)]) 
    
    Linear_R2_matrix[ar - ar_min, :] = Linear_R2_list
    Linear_RMSE_matrix[ar - ar_min, :] = Linear_RMSE_list
    
    '''========================================================================================================'''
    '''============================DNN model for 50 steps ahead prediction=================================='''
    '''========================================================================================================'''
    tr_X, tr_Y = Lor_AR_dat[:int(0.7*len(Lor_AR_dat)), :3*ar], Lor_AR_dat[:int(0.7*len(Lor_AR_dat)), 3*ar:(3*(ar+1))]
    te_X, te_Y = Lor_AR_dat[int(0.7*len(Lor_AR_dat)):, :3*ar], Lor_AR_dat[int(0.7*len(Lor_AR_dat)):, 3*ar:(3*(ar+1))]
    te_Y_full = Lor_AR_dat[int(0.7*len(Lor_AR_dat)):, 3*ar:]
    
    tr_X, tr_Y = Variable(torch.from_numpy(tr_X)),Variable(torch.from_numpy(tr_Y))
    
    # another way to define a network
    net = torch.nn.Sequential(
            torch.nn.Linear(3*ar, 70),
            torch.nn.ReLU(),
            torch.nn.Linear(70, 3)
        )
    
    net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.04, weight_decay = 0.00001)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    
    torch_dataset = Data.TensorDataset(tr_X, tr_Y)
    loader = Data.DataLoader(
        dataset=torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=2,)
    
    #------Training the model------#
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader): # for each training step
            b_x = Variable(batch_x.type(torch.cuda.FloatTensor))
            b_y = Variable(batch_y.type(torch.cuda.FloatTensor))
            prediction = net(b_x.float())     # input x and predict based on x
            loss = loss_func(prediction, b_y.reshape(prediction.shape).float())     # must be (1. nn output, 2. target)
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
     
        val_pred = net(Variable(torch.from_numpy(te_X).type(torch.cuda.FloatTensor)))
#        print(epoch)
#        print(calculate_RMSE(te_Y, val_pred.cpu().detach().numpy()))
#        print(metrics.r2_score(te_Y, val_pred.cpu().detach().numpy()))
    
    #------------Generating the k-step ahead prediction--------------#
    Prediction_Final =  val_pred.cpu().detach().numpy().copy()
    te_temp = te_X.copy()
    
    for i in range(step_pred_max):
        val_pred0 = net(Variable(torch.from_numpy(te_X).type(torch.cuda.FloatTensor)))
        val_pred0 = val_pred0.cpu().detach().numpy()
        Prediction_Final = np.concatenate((Prediction_Final, val_pred0), axis = 1)
        te_X = np.roll(te_X, -3, axis = 1)
        te_X[:, (3*(ar-1)):(3*ar)] = val_pred0
    
    
    Prediction_Final = Prediction_Final[:, 3:]
        
    DNN_R2_list = np.array([metrics.r2_score(te_Y_full[:, i:(i+3)], Prediction_Final[:, i:(i+3)]) for i in np.arange(0, step_pred_max*3, 3)]) 
    DNN_RMSE_list = np.array([calculate_RMSE(te_Y_full[:, i:(i+3)], Prediction_Final[:, i:(i+3)]) for i in np.arange(0, step_pred_max*3, 3)]) 
    
    DNN_R2_matrix[ar - ar_min, :] = DNN_R2_list
    DNN_RMSE_matrix[ar - ar_min, :] = DNN_RMSE_list
    
np.save('.\Out_Data\LO_Linear_R2.npy', Linear_R2_matrix)
np.save('.\Out_Data\LO_Linear_RMSE.npy', Linear_RMSE_matrix)
np.save('.\Out_Data\LO_DNN_R2.npy', DNN_R2_matrix)
np.save('.\Out_Data\LO_DNN_RMSE.npy', DNN_RMSE_matrix)

Linear_R2_matrix = np.load('.\Out_Data\LO_Linear_MC_R2.npy')
Linear_RMSE_matrix = np.load('.\Out_Data\LO_Linear_MC_RMSE.npy')
DNN_R2_matrix = np.load('.\Out_Data\LO_DNN_MC_R2.npy')
DNN_RMSE_matrix = np.load('.\Out_Data\LO_DNN_MC_RMSE.npy')

display_plot(np.clip(Linear_R2_matrix, 0, 1), size = (30, 18), metric = 'R2', title = 'Lorenz LINEAR', dst_name = '.\Out_Data\Linear_LO_R2.png', save_flg = 1)
display_plot(Linear_RMSE_matrix, size = (30, 18), metric = 'RMSE', title = 'Lorenz LINEAR', dst_name = '.\Out_Data\Linear_LO_RMSE.png', save_flg = 1)
display_plot(np.clip(DNN_R2_matrix, 0, 1), size = (30, 18), metric = 'R2', title = 'Lorenz DNN', dst_name = '.\Out_Data\DNN_LO_R2.png', save_flg = 1)
display_plot(DNN_RMSE_matrix, size = (30, 18), metric = 'RMSE', title = 'Lorenz DNN', dst_name = '.\Out_Data\DNN_LO_RMSE.png', save_flg = 1)    