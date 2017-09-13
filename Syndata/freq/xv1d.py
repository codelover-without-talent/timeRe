'''
Created on Sep 1, 2017

@author: michael
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools
import multiprocessing as mp


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

n_tr = 500
n_tt = 500


# variance for the noise on the label
sigma_label = 0.01

eta0 = 1.0

np.random.seed(2017)

# the variance for the marginal distribution
sigma_tr = np.random.uniform(0,5,l_tr)
sigma_tt = np.random.uniform(0,5,l_tt)


rho_tr = np.random.uniform(0.5,1,l_tr)
rho_tt = np.random.uniform(0.5,1,l_tt)


#### 1d data
# correlation parameter for ar1  1d

rho_tr,sample_tr,sample_statn_tr,entro_tr,entro_tr_label = dg.cor_sample_gen(l_tr, n_tr,sigma_tr, rho_tr,sigma_label)
rho_tt,sample_tt,sample_statn_tt,entro_tt,entro_tt_label = dg.cor_sample_gen(l_tt, n_tt, sigma_tt,rho_tt,sigma_label)



 
# #define grid 
np.random.seed()
theta_sml_grid = np.random.uniform(0,20,20)
theta_bag_grid = np.random.uniform(0,20,20)
lmba_grid = 10.0**(np.arange(-8,2,1))
par_grid0 = list(itertools.product(theta_sml_grid,theta_sml_grid,lmba_grid))



NumFolds0 = 5
n_bag0 = 2
processes0 = 10




def xval_disRe_corc1_processes(processes,par_grid,eta,n_bag,X,Y,NumFolds):
    pool = mp.Pool(processes = processes)
    results = [pool.apply_async(kts.xval_disRe_corc1,args=(par,eta,n_bag,X,Y,NumFolds)) for par in par_grid]
    results = [p.get() for p in results]
    return results

def xval_disRe_processes(processes,par_grid,n_bag,X,Y,NumFolds):
    pool = mp.Pool(processes = processes)
    results = [pool.apply_async(kts.xval_disRe,args=(par,n_bag,X,Y,NumFolds)) for par in par_grid]
    results = [p.get() for p in results]
    return results

results_hat = xval_disRe_processes(processes0,par_grid0,n_bag0,sample_tr,entro_tr_label,NumFolds0)
results_hat_dict = results_hat[0]
l_hat = len(results_hat)
for ii in np.arange(1,l_hat):
    results_hat_dict.update(results_hat[ii])
rmse_hat = results_hat_dict.keys()
xv_hat = results_hat_dict.values()
min_rmse_hat = np.argmin(rmse_hat)

par_xv_hat = xv_hat[min_rmse_hat]


 
results_bar = xval_disRe_corc1_processes(processes0,par_grid0,eta0,n_bag0,sample_tr,entro_tr_label,NumFolds0)
results_bar_dict = results_bar[0]
l_bar = len(results_bar)
for ii in np.arange(1,l_bar):
    results_bar_dict.update(results_bar[ii])
rmse_bar = results_bar_dict.keys()
xv_bar = results_bar_dict.values()
min_rmse_bar = np.argmin(rmse_bar)

par_xv_bar = xv_bar[min_rmse_bar]
 

results_statn = xval_disRe_processes(processes0,par_grid0,n_bag0,sample_statn_tr,entro_tr_label,NumFolds0)
results_statn_dict = results_statn[0]
l_statn = len(results_statn)
for ii in np.arange(1,l_statn):
    results_statn_dict.update(results_statn[ii])
rmse_statn = results_statn_dict.keys()
xv_statn = results_statn_dict.values()
min_rmse_statn = np.argmin(rmse_statn)

par_xv_statn = xv_hat[min_rmse_statn]






# save computation
pickle_out1 = open("xv for plain model","wb")
pickle_out2 = open ("xv for stationary dist", "wb")
pickle_out3 = open("xv for corrected model","wb")
    
pickle.dump(par_xv_hat,pickle_out1)
pickle.dump(par_xv_statn,pickle_out2)
pickle.dump(par_xv_bar,pickle_out3)
    
pickle_out1.close()
pickle_out2.close()
pickle_out3.close()
 
 
  
  
# perform dist. regression after cross validation
lnd_den = kts.lnd_choe(n_bag0,sample_tr)
lnd_ind = kts.lnd_choe(n_bag0,sample_statn_tr)
print lnd_den.shape
pickle_out4 = open("landmark for depen","wb")
pickle_out5 = open("landmark for statn","wb")
 
pickle.dump(lnd_den,pickle_out4)
pickle.dump(lnd_ind,pickle_out5)
 
pickle_out4.close()
pickle_out5.close()
 
 

# regression on dependent data
beta_hat,prdt_hat,err_hat = kts.disRe(par_xv_hat,lnd_den,sample_tr,entro_tr_label,sample_tt,entro_tt_label)


 
# regression on corrected model
beta_bar,prdt_bar,err_bar = kts.disRe_corc1(par_xv_bar,eta0,lnd_den,sample_tr,entro_tr_label,sample_tt,entro_tt_label)
   

   
# regression on independent data
beta_statn,prdt_statn,err_statn = kts.disRe(par_xv_statn,lnd_ind,sample_statn_tr,entro_tr_label,sample_statn_tt,entro_tt_label)
   
# ybar
prdt_ybar = np.mean(entro_tr_label)
err_ybar = np.sqrt(np.mean((prdt_ybar-entro_tt_label)**2))
 
print err_hat
print err_bar
print err_statn
print err_ybar
 
 
# # #plot
fig = plt.figure()
# # #                        
axes = plt.gca()
   
rho_index = np.argsort(rho_tt)
plt.plot(rho_tt[rho_index],entro_tt[rho_index],c = 'k',label = 'True Entropy')
plt.plot(rho_tt[rho_index],prdt_bar[rho_index],color = "r",linestyle = "-.",marker = ".",label = 'Correction')
plt.plot(rho_tt[rho_index],prdt_hat[rho_index],color = "b",linestyle = "-.",marker = ".",label = 'No Correction')
plt.plot(rho_tt[rho_index],prdt_statn[rho_index],c='g',linestyle = "-.",label = "Stationary  Model")
plt.plot(rho_tt[rho_index],np.repeat(prdt_ybar,len(rho_tt)),color='y',linestyle = "-.",label = "Sample Mean")
plt.xlabel("rho")
plt.ylabel("Entropy")
#plt.savefig("model comp1d.pdf")
plt.show() 

