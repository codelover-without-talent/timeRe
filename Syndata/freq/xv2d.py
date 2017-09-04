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
  
# #define grid 
np.random.seed()
theta_sml_grid = np.random.uniform(0,20,15)
theta_bag_grid = np.random.uniform(0,20,15)
lmba_grid = 10.0**(np.arange(-8,2,1))
NumFolds = 5
n_bag = 5
   
rmse_hat,aver_hat,par_xv_hat = kts.xval_disRe(theta_sml_grid,theta_bag_grid,lmba_grid,n_bag,ar_tr,entro_tr_label,NumFolds)
#   
   
rmse_statn,aver_statn,par_xv_statn = kts.xval_disRe(theta_sml_grid,theta_bag_grid,lmba_grid,n_bag,ar_statn_tr,entro_tr_label,NumFolds)


print par_xv_hat
print par_xv_statn
 
# save computation
pickle_out1 = open("2d xv for plain model","wb")
pickle_out2 = open ("2d xv for stationary dist", "wb")
#pickle_out3 = open("ridge_rff error","wb")
   
pickle.dump(par_xv_hat,pickle_out1)
pickle.dump(par_xv_statn,pickle_out2)
#pickle.dump(mse_pre_rff,pickle_out3)
   
pickle_out1.close()
pickle_out2.close()
#pickle_out3.close()


 
 
# perform dist. regression after cross validation
lnd_den = kts.lnd_choe(n_bag,ar_tr)
lnd_ind = kts.lnd_choe(n_bag,ar_statn_tr)

pickle_out3 = open("2d landmark for depen","wb")
pickle_out4 = open("2d landmark for statn","wb")

pickle.dump(lnd_den,pickle_out3)
pickle.dump(lnd_ind,pickle_out4)

pickle_out3.close()
pickle_out4.close()
  
# regression on dependent data
beta_hat,prdt_hat,err_hat = kts.disRe(par_xv_hat,lnd_den,ar_tr,entro_tr_label,ar_tt,entro_tt_label)
  
# regression on independent data
beta_statn,prdt_statn,err_statn = kts.disRe(par_xv_statn,lnd_ind,ar_statn_tr,entro_tr_label,ar_statn_tt,entro_tt_label)
  
# ybar
prdt_ybar  =np.mean(entro_tr_label)
err_ybar = np.sqrt(np.mean((prdt_ybar-entro_tt_label)**2))

print err_hat
print err_statn
