# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 01:34:08 2019

@author: sayon
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.special import comb

def g_k(n, k, mu):
    return comb(n-1, k-1)*np.power(mu, k)*np.power(1-mu, n-k)*float(n>=k)

# Generate the X_k[n] variable for X[n] by convolving X[n]*G_k[n] and return the X_k[n] variable 
def generate_Gamma_filter_coeff(n = 50, K = 10, mu = 0.5, k_query = 1, dst_name = ''):
    G_K_filter_matrix = np.zeros((n, 1))
    G_K_filter_matrix[0, 0] = 1.0
    
    for k in np.arange(1, K+1):
        vfunc = np.vectorize(g_k)
        G_k_filter = vfunc(n = np.arange(k, n), k = k, mu = mu)
        G_k_filter = np.append(np.zeros(k), G_k_filter)
        G_k_filter = G_k_filter.reshape((len(G_k_filter), 1))
        G_K_filter_matrix = np.concatenate((G_K_filter_matrix, G_k_filter), axis = 1)

    plt.figure(figsize= (10, 5))    
    plt.plot(G_K_filter_matrix[:, k_query], '-')
    plt.ylabel('h [n]')
    plt.xlabel('n')
    plt.title('Impulse Response')
#    plt.savefig(dst_name)
    return G_K_filter_matrix

def Laguarre_response(signal, b = torch.tensor(0.0), K_max = 5):
    result = torch.zeros(signal.shape[0]+1, K_max)
    one_minus_b_sq = torch.sqrt(1-b**2)
    for i in range(1, result.shape[0]):
        result[i, 0] = b*result[i-1, 0] + one_minus_b_sq*signal[i-1, 0]
    
    for n in range(1, result.shape[0]):
        for k in range(1, K_max):
            result[n, k] = b*(result[n-1, k] - result[n, k-1]) + result[n-1, k-1]
    result = result[1:, :]        
    return result

signal = torch.zeros(150, 1)
signal[0, 0] = 1.0
res = Laguarre_response(signal, b = torch.tensor(0.9), K_max = 5)


#res = generate_Gamma_filter_coeff(n = 30, K = 3, mu = 1.8, k_query = 3, dst_name = 'D:\workspace\pytorch\Lorenz_Mackey_DNN\Results\Gamma_mu01_k01.png')

K_array = np.arange(0, 5)

plt.figure(figsize= (15, 10))    
labels = ["k = "+str(i) for i in np.arange(0, 5)]
for i in K_array:    
    plt.plot(res[:, i], label = labels[i-1])
plt.rcParams.update({'font.size': 15})
plt.legend()
plt.ylabel('h [n]')
plt.xlabel('n')
plt.title('Impulse Response of Laguerre filter - b = 0.9')

#plt.title('Impulse Response of Gamma filter - mu = 1.8')

plt.savefig('D:\workspace\pytorch\Lorenz_Mackey_DNN\Results\Laguerre_basis_b09.png')

res.sum(axis = 0)
