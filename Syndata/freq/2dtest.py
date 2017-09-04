'''
Created on Sep 4, 2017

@author: michael
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pickle


from kerpy.Kernel import Kernel
from kerpy.BagKernel import BagKernel
from kerpy.GaussianKernel import GaussianKernel
from kerpy.GaussianBagKernel import GaussianBagKernel
#import pandas as pd
import scipy.stats as sps
import scipy.spatial.distance as spd

import kerpy.time_series as kts
import data_generation as dg

### define parameters
l_tr = 100
l_tt = 25

n_tr = 200
n_tt = 200

# variance for the noise in ar 1
sigma0 = 0.01
# variance for the noise on the label
sigma_label = 0.01



# 2d data
np.random.seed(2010) 
#eigenvalue for trasition matrix in ar 1 process 2d

lmba1_tr = np.random.uniform(-1,1,1)
lmba2_tr = np.random.uniform(-1,1,1)    
lmba1_tt =lmba1_tr# np.random.uniform(-1,1,1)
lmba2_tt = lmba2_tr# np.random.uniform(-1,1,1) 
  
alpha_tr,ar_tr,ar_statn_tr,entro_tr,entro_tr_label = dg.sam2d_gen(lmba1_tr,lmba2_tr,l_tr,n_tr,sigma0,sigma_label)
alpha_tt,ar_tt,ar_statn_tt,entro_tt,entro_tt_label = dg.sam2d_gen(lmba1_tt,lmba2_tt,l_tt,n_tt,sigma0,sigma_label)


par_xv_hat = pickle.load(open("2d xv for plain model","rb"))
par_xv_statn = pickle.load(open("2d xv for stationary dist","rb"))
print par_xv_hat
print par_xv_statn


lnd_den = pickle.load(open("2d landmark for depen","rb"))
lnd_ind = pickle.load(open("2d landmark for statn","rb"))

# regression on dependent data
beta_hat,prdt_hat,err_hat = kts.disRe(par_xv_hat,lnd_den,ar_tr,entro_tr_label,ar_tt,entro_tt_label)
  
# regression on independent data
beta_statn,prdt_statn,err_statn = kts.disRe(par_xv_statn,lnd_ind,ar_statn_tr,entro_tr_label,ar_statn_tt,entro_tt_label)
  
# ybar
prdt_ybar  =np.mean(entro_tr_label)
err_ybar = np.sqrt(np.mean((prdt_ybar-entro_tt_label)**2))

print err_hat
print err_statn
print err_ybar
 
# # #plot
fig = plt.figure()
# # #                        
axes = plt.gca()
  
alpha_index = np.argsort(alpha_tt)
plt.plot(alpha_tt[alpha_index],entro_tt[alpha_index],c = 'k',label = 'True Entropy')
plt.plot(alpha_tt[alpha_index],prdt_hat[alpha_index],color = "b",linestyle = "-.",marker = ".",label = 'No Correction')
plt.plot(alpha_tt[alpha_index],prdt_statn[alpha_index],c='g',linestyle = "-.",label = "Stationary  Model")
plt.plot(alpha_tt[alpha_index],np.repeat(prdt_ybar,len(alpha_tt)),color='y',linestyle = "-.",label = "Sample Mean")
plt.xlabel("rho")
plt.ylabel("True Entropy")
plt.savefig("model comp1.pdf")
plt.show() 

