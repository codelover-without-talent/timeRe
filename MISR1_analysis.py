'''
Created on Dec 26, 2017

@author: michael
'''
from __future__ import division
import numpy as np
import scipy.stats as sps
import scipy.spatial.distance as spd
import multiprocessing as mp
import pickle
import csv

from kerpy.Kernel import Kernel
from kerpy.BagKernel import BagKernel
from kerpy.GaussianKernel import GaussianKernel
from kerpy.GaussianBagKernel import GaussianBagKernel

import itertools

import kerpy.cor_dist_reg as kdr


#----------------------------------------------------------------------
### read data
bag_num0 = []
inst0 = []
label0 = []
def csv_reader(file_obj):
    """
    Read a csv file
    """
    reader = csv.reader(file_obj)
    for row in reader:
        #print(" ".join(row))
        x0 = row
        bag_num = int(row[0])
        ins = row[1:17]
        inss = [float(x) for x in ins]
        insss = np.array(inss)
        label = float(row[17])
        
        bag_num0.append(bag_num)
        inst0.append(insss)
        label0.append(label)
        
#----------------------------------------------------------------------
if __name__ == "__main__":
    csv_path = "misr1.csv"
    with open(csv_path, "rb") as f_obj:
        misr1 = csv_reader(f_obj)


### convert data into the correct format
index1 = np.arange(0,80100,100)
index1_len = len(index1)


 
sample_matrix = np.zeros((index1[-1],len(inst0[0])))
for jj in range(index1[-1]):
    sample_matrix[jj,:] = inst0[jj] 

samp_gp0 = []
label_gp0 = []
for ii in range(1,index1_len):
    num_start = index1[ii-1] 
    num_end = index1[ii]
    
    sample = sample_matrix[num_start:num_end,:]

    samp_gp0.append(sample)
    
    label = label0[num_start]
    label_gp0.append(label)
#print len(samp_gp0)



### split data for training sample and testing sample
NumFolds_test = 5
trn_samp, trn_label, tst_samp,tst_label = kdr.tr_tst_split(samp_gp0,label_gp0,NumFolds_test)


#######################
# our training sample #
#######################
samp_train = trn_samp[0]
label_train = trn_label[0]


###################### 
# our testing sample #
######################
samp_test = tst_samp[0]
label_test = tst_label[0]

data_in_use = [samp_train,label_train,samp_test,label_test]
### split data for cross validation
NumFolds_cv = 4



### define parallel function for 
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


lnd_bag_xval0 = 2
lnd_bag_num = 200
samp_lnd_chic = samp_train[:lnd_bag_num]

lnd_mk_den0 = kdr.lnd_choe(lnd_bag_xval0,samp_lnd_chic)

processes = 20
 

 
 
emp_xval_results0 = emp_statn_xval_processes(processes,par_grid0,lnd_mk_den0,samp_train,label_train,NumFolds_cv)  
emp_error_keys = []
emp_error_vals = []
for dic in emp_xval_results0:
    key0 = dic.keys()
    val0 = dic.values()
         
    emp_error_keys.append(key0)
    emp_error_vals.append(val0)
         
     
emp_min_error_pos = emp_error_keys.index(min(emp_error_keys))
emp_xval_par = emp_error_vals[emp_min_error_pos][0]
 
 
 
    
    
shrin_xval_results0 = shrin_xval_processes(processes,par_grid0,lnd_mk_den0,samp_train,label_train,NumFolds_cv)
shrin_error_keys = []
shrin_error_vals = []
for dic in shrin_xval_results0:
    key0 = dic.keys()
    val0 = dic.values()
          
    shrin_error_keys.append(key0)
    shrin_error_vals.append(val0)
          
      
shrin_min_error_pos = shrin_error_keys.index(min(shrin_error_keys))
shrin_xval_par = shrin_error_vals[shrin_min_error_pos][0]
 

 
 
par_xval = [emp_xval_par,shrin_xval_par]
 
pickle_out1 = open("sample_data_misr","wb")
pickle.dump(data_in_use, pickle_out1)
pickle_out1.close()
       
pickle_out2 = open("lnd_points_misr","wb")
pickle.dump(lnd_mk_den0, pickle_out2)
pickle_out2.close()
  
pickle_out3 = open("parameter_cv_misr","wb")
pickle.dump(par_xval, pickle_out3)
pickle_out3.close()
