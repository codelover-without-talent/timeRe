'''
Created on Sep 2, 2017

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
#np.random.seed(2010)
rho_tr,sample_tr,sample_statn_tr,entro_tr,entro_tr_label = dg.cor_sample_gen(l_tr, n_tr,sigma_tr, rho_tr,sigma_label)
rho_tt,sample_tt,sample_statn_tt,entro_tt,entro_tt_label = dg.cor_sample_gen(l_tt, n_tt, sigma_tt,rho_tt,sigma_label)





par_xv_hat = pickle.load(open("xv for plain model","rb"))
par_xv_bar = pickle.load(open("xv for corrected model","rb"))
par_xv_statn = pickle.load(open("xv for stationary dist","rb"))


lnd_den = pickle.load(open("landmark for depen","rb"))
lnd_ind = pickle.load(open("landmark for statn","rb"))




# regression on dependent data
beta_hat,prdt_hat,err_hat = kts.disRe(par_xv_hat,lnd_den,sample_tr,entro_tr_label,sample_tt,entro_tt_label)
  

# regression on corrected model
beta_bar,prdt_bar,err_bar = kts.disRe_corc1(par_xv_bar,eta0,lnd_den,sample_tr,entro_tr_label,sample_tt,entro_tt_label)

# regression on independent data
beta_statn,prdt_statn,err_statn = kts.disRe(par_xv_statn,lnd_ind,sample_statn_tr,entro_tr_label,sample_statn_tt,entro_tt_label)
   
# ybar
prdt_ybar  = np.mean(entro_tr_label)
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
plt.ylabel("True Entropy")
#plt.savefig("model comp1d.pdf")
plt.show() 

