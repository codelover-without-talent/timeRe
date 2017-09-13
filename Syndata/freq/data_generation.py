from __future__ import division
import numpy as np




########## one dimensional data
# generating single ar 1 process
def uni_ar1(phi,sigma,num):
	ar1 = np.zeros((num,1))
	
	ar1[0] = np.random.normal(0,np.sqrt(sigma),1)
	
	for ii in np.arange(1,num):
		ar1[ii,0] = phi*ar1[ii-1] + np.random.normal(0,np.sqrt(sigma),1)
	entro1 = 0.5 * np.log(2*np.pi*np.e*sigma/(1-phi**2))
	
	return ar1,entro1
   
   
   
# generating bags of ar 1 process
def sam_gen(phi,l,num,sigma,sigma0):
    ar_sam = list()
    ar_statn_sam = list()
    entro_sam = np.zeros(l)
    entro_sam_var = np.zeros(l)
    
    for ii in np.arange(l):
        phi_ar = phi[ii]
        statn_variance = sigma/(1-phi_ar**2)
        ar0,entro0 = uni_ar1(phi_ar,sigma,num)
		
        ar_statn0 = np.random.normal(0,np.sqrt(statn_variance),num).reshape((num,1))
          
        entro_sam[ii] = entro0
		
        entro_sam_var[ii] = entro0 + np.random.normal(0,np.sqrt(sigma0),1)
        ar_sam.append(ar0)
        ar_statn_sam.append(ar_statn0)
    return ar_sam,ar_statn_sam,entro_sam,entro_sam_var

####################### 2 dimensional data
def ar_2d(lmbda1,lmbda2,alpha,sigma,num):
    a0 = lmbda1#np.random.uniform(-1,1,1)
    b0 = lmbda2#np.random.uniform(-1,1,1)
    sigma_matrix = sigma * np.eye(2)
       
       
    ab0 =np.diag(np.ravel(np.array([a0,b0])))
       
    rot_m = np.array([[np.cos(alpha),np.sin(alpha)],[np.sin(alpha),-np.cos(alpha)]])
    tran_m = rot_m.dot(ab0).dot(rot_m)
       
          
    sigma_statn = np.zeros((2,2))
    sigma_statn[0,0] = sigma* (1/(1-a0**2)* np.cos(alpha)**2 + 1/(1-b0**2) * np.sin(alpha)**2)
    sigma_statn[0,1] = sigma*(1/(1-a0**2)* np.cos(alpha)*np.sin(alpha)-1/(1-b0**2)* np.sin(alpha)*np.cos(alpha))
    sigma_statn[1,0] = sigma*(1/(1-a0**2)* np.cos(alpha)*np.sin(alpha)-1/(1-b0**2)* np.sin(alpha)*np.cos(alpha)) 
    sigma_statn[1,1] = sigma* (1/(1-a0**2)* np.sin(alpha)**2 + 1/(1-b0**2) * np.cos(alpha)**2)
       
           
       
    ar2_statn = np.random.multivariate_normal(np.array([0,0]),sigma_statn,num)
       
    ar2 = np.zeros((num,2))
    ar2[0,:]=np.random.multivariate_normal(np.array([0,0]),sigma_matrix,1)
       
    for ii in np.arange(1,num):
        ar2[ii,:] = tran_m.dot(ar2[ii-1,:])+np.random.multivariate_normal(np.array([0,0]),sigma_matrix,1)
       
    var0 =sigma* 1/(1-a0**2)*(np.cos(alpha)**2)+1/(1-b0**2)*(np.sin(alpha)**2)
    entro2 = 0.5 * np.log(2*np.pi*np.e*var0)
       
    return ar2,ar2_statn,entro2
   
def sam2d_gen(lmbda1,lmbda2,l,num,sigma_2d,sigma0):
    alpha = np.random.uniform(0,2*np.pi,l) #np.repeat(0.5*np.pi,l)  
    ar_sam = list()
    ar_statn_sam = list()
    entro_sam = np.zeros(l)
    entro_sam_var = np.zeros(l)
       
    for ii in np.arange(l):
        alpha0 = alpha[ii]
        ar0,ar_statn0,entro0 = ar_2d(lmbda1,lmbda2,alpha0,sigma_2d,num)
           
        entro_sam[ii] = entro0
        entro_sam_var[ii] = entro0 + np.random.normal(0,np.sqrt(sigma0),1)
        ar_sam.append(ar0)
        ar_statn_sam.append(ar_statn0)
   
       
    return alpha,ar_sam,ar_statn_sam, entro_sam,entro_sam_var
#########
# draw samples from a high dimensional multivariate normal

#define the covariance matrix for the high dimensional gaussian distribution   
def multi_cov(n,rho,sigma):
    mat_cov = np.eye(n)
    for ii in np.arange(n):
        for jj in np.arange(n):
            if ii-jj != 0:
                mat_cov[ii,jj] = rho**abs(ii-jj)
    return sigma**2 * mat_cov

def cor_sample(n,rho,sigma):
	cov = multi_cov(n,rho,sigma)
	sample = np.random.multivariate_normal(np.zeros(n),cov)
	
	return sample

def cor_sample_gen(l,n,sigma,rho,sigma_label):
	samples = list()
	samples_statn = list()
	entro = np.ones(l)*0.5 * np.log(2*np.pi*np.e*sigma)
	entro_label = np.zeros(l)
	
	
	for ii in np.arange(l):
		sample = cor_sample(n,rho[ii],sigma[ii]).reshape((n,1))
		sample_statn = np.random.normal(0,sigma[ii],n).reshape((n,1))
		entro_label[ii] = entro[ii] + np.random.normal(0,sigma_label,1)
		
		samples.append(sample)
		samples_statn.append(sample_statn)
	return rho,samples,samples_statn,entro,entro_label

