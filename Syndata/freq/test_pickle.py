'''
Created on Sep 2, 2017

@author: michael
'''
import pickle

par_xv_hat = pickle.load(open("xv for plain model","rb"))
par_xv_statn = pickle.load(open("xv for stationary dist","rb"))

print par_xv_hat
print par_xv_statn