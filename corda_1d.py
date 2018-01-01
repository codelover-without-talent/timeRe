'''
Created on Nov 20, 2017

@author: michael
'''
from __future__ import division
import numpy as np
import scipy.stats as sps
import scipy.spatial.distance as spd
import multiprocessing as mp
import pickle
import matplotlib.pyplot as plt

from kerpy.Kernel import Kernel
from kerpy.BagKernel import BagKernel
from kerpy.GaussianKernel import GaussianKernel
from kerpy.GaussianBagKernel import GaussianBagKernel

import itertools

import kerpy.cor_dist_reg as kdr

l_tr = 100
l_tt = 25

n_sample = 200
rho = 0.4

sigma_label = 0.3



sigma_tr = np.random.uniform(0,1,l_tr)
samp0,samp_statn0,entro0,en_lab0 = kdr.cor_sample_gen(l_tr,n_sample,rho,sigma_tr,sigma_label)

sample_data_tr = [samp0,samp_statn0,entro0,en_lab0]

  
sigma_tt = np.random.uniform(0,1,l_tt)
samp1,samp_statn1,entro1,en_lab1 = kdr.cor_sample_gen(l_tt,n_sample,rho,sigma_tt,sigma_label)
sample_data_tt = [samp1,samp_statn1,entro1,en_lab1,sigma_tt]



def emp_statn_xval_processes(processes,par_list,lnd_mk,X,Y,NumFolds):
    pool = mp.Pool(processes = processes)
    results = [pool.apply_async(kdr.emp_statn_xval,args=(par,lnd_mk,X,Y,NumFolds)) for par in par_list]
    results = [p.get() for p in results]
    return results
def shrin_xval_processes(processes,par_list,lnd_mk,X,Y,NumFolds):
    pool = mp.Pool(processes = processes)
    results = [pool.apply_async(kdr.shrin_xval,args = (par,lnd_mk,X,Y,NumFolds)) for par in par_list]
    results = [p.get() for p in results]
    return results


theta_data_grid = np.exp(-13+np.arange(25))
 
lmba_grid = np.exp(np.arange(-15,0,0.5))
 
par_grid0 = list(itertools.product(theta_data_grid,lmba_grid))


for ii in np.arange(len(par_grid0)):
    par_gridi = list(par_grid0[ii])
    prior_theta0 = par_gridi[0]*8.0
    par_gridi.insert(0,prior_theta0)
    par_grid0[ii] = tuple(par_gridi)


lnd_bag_xval0 = 3
lnd_mk_den0 = kdr.lnd_choe(lnd_bag_xval0,samp0)
lnd_mk_ind0 = kdr.lnd_choe(lnd_bag_xval0,samp_statn0)
lnd_pot0 = [lnd_mk_den0,lnd_mk_ind0]

NumFolds = 5
processes = 20






 
emp_xval_results0 = emp_statn_xval_processes(processes,par_grid0,lnd_mk_den0,samp0,entro0,NumFolds)  
emp_error_keys = []
emp_error_vals = []
for dic in emp_xval_results0:
    key0 = dic.keys()
    val0 = dic.values()
        
    emp_error_keys.append(key0)
    emp_error_vals.append(val0)
        
    
emp_min_error_pos = emp_error_keys.index(min(emp_error_keys))
emp_xval_par = emp_error_vals[emp_min_error_pos][0]

   
   
shrin_xval_results0 = shrin_xval_processes(processes,par_grid0,lnd_mk_den0,samp0,entro0,NumFolds)
shrin_error_keys = []
shrin_error_vals = []
for dic in shrin_xval_results0:
    key0 = dic.keys()
    val0 = dic.values()
        
    shrin_error_keys.append(key0)
    shrin_error_vals.append(val0)
        
    
shrin_min_error_pos = shrin_error_keys.index(min(shrin_error_keys))
shrin_xval_par = shrin_error_vals[shrin_min_error_pos][0]



statn_xval_results0 = emp_statn_xval_processes(processes,par_grid0,lnd_mk_ind0,samp_statn0,entro0,NumFolds)
statn_error_keys = []
statn_error_vals = []
for dic in statn_xval_results0:
    key0 = dic.keys()
    val0 = dic.values()
        
    statn_error_keys.append(key0)
    statn_error_vals.append(val0)
        
    
statn_min_error_pos = statn_error_keys.index(min(statn_error_keys))
statn_xval_par = statn_error_vals[statn_min_error_pos][0]


par_xval = [emp_xval_par,shrin_xval_par,statn_xval_par]



 
 
pickle_out1 = open("sample_training_data","wb")
pickle.dump(sample_data_tr, pickle_out1)
pickle_out1.close()
      
pickle_out2 = open("sample_testing_data","wb")
pickle.dump(sample_data_tt, pickle_out2)
pickle_out2.close()
 
pickle_out3 = open("parameter_crossvalidated","wb")
pickle.dump(par_xval, pickle_out3)
pickle_out3.close()
 
pickle_out4 = open("landmark_point","wb")
pickle.dump(lnd_pot0, pickle_out4)
pickle_out4.close()
      
     
