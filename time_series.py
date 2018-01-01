from __future__ import division
import numpy as np
import scipy.stats as sps
import scipy.spatial.distance as spd

# define the covariance matrix for the GP process with mean embedding
def smooth_sqkernel(theta, eta,X,Y=None):
    """
    Computes r, the kernel for the mean embedding 
    
    X - 2d numpy.ndarray, first set of samples:
        number of rows: number of samples
        number of columns: dimensionality
        Y - 2d numpy.ndarray, second set of samples, can be None in which case its replaced by X
    """
    assert(len(np.shape(X))==2)
    
    #D = float(np.shape(X)[1])
        
    # if X=Y,use more efficient pdist call which exploits symmetry
    if Y is None:
       sq_dists = spd.squareform(spd.pdist(X, 'sqeuclidean'))
       sq_dists_extra = spd.squareform(spd.pdist(X,"sqeuclidean"))+ 4* np.dot(X,X.T)
    else:
        #GenericTests.check_type(Y, 'Y',np.ndarray)
        assert(len(np.shape(Y))==2)
        assert(np.shape(X)[1]==np.shape(Y)[1])
        sq_dists = spd.cdist(X, Y, 'sqeuclidean')
        sq_dists_extra = spd.cdist(X, Y, 'sqeuclidean')+ 4 * np.dot(X,Y.T)
    K = np.exp(-1/4 * 1/theta * sq_dists) * np.exp(-1/8 *1/(theta/2 + eta**2) * sq_dists_extra) #(2* np.pi)**(D/2) * (2/theta + 1/(eta**2))**(-0.5 * D) * 
    return K

# define the squared exponential kernel
def sqkernel(theta,X, Y=None):
    """
    Computes r, the kernel for the mean embedding 
    
    X - 2d numpy.ndarray, first set of samples:
        number of rows: number of samples
        number of columns: dimensionality
        Y - 2d numpy.ndarray, second set of samples, can be None in which case its replaced by X
    """
    assert(len(np.shape(X))==2)
    
    # if X=Y, use more efficient pdist call which exploits symmetry
    if Y is None:
       sq_dists = spd.squareform(spd.pdist(X, 'sqeuclidean'))
       
    else:
        #GenericTests.check_type(Y, 'Y',np.ndarray)
        assert(len(np.shape(Y))==2)
        assert(np.shape(X)[1]==np.shape(Y)[1])
        sq_dists = spd.cdist(X, Y, 'sqeuclidean')
        
    K = np.exp(-1/(2 * theta) * sq_dists)
    return K

# define the aotuocovariance function for any two vectors    
def autocov(theta,X,Y):
    D = np.shape(X)[0]
    Ks = sqkernel(theta,X)
    K_s = sqkernel(theta,Y)
    Hs =np.eye(D)-1/D * np.outer(np.ones(D),np.ones(D))
    gamma_s0 = D**(-2) * np.trace(Ks.dot(Hs).dot(K_s).dot(Hs))
    gamma_s = np.sqrt(gamma_s0)
     
    return gamma_s    


# compute effective variance of a given time series
def var_ts(theta,X,m):
    """
    we are computing the variance for the time series for each bag,
    here theta is the parameter for the kernel,
    m is the size of the number of z points we use
    """
    n = X.shape[0]
    nn = int(n/2)
    var0 = np.zeros(nn)
    for ii in np.arange(nn):
        index2 = np.arange(ii)
        index1 = np.arange(n-ii,n)
        
        xs = np.delete(X,index1,0)
        x_s = np.delete(X,index2,0)
        
        var0[ii] =2*(1-ii/n) * autocov(theta,xs,x_s)
    
    var0[0] = var0[0]/2.0 
    var1 = np.mean(var0)* np.eye(m) 
    
    return var1
#function to test whether use only one bag of sample would lead to bias in estimating auto-covariance
def var_test(theta,X,m):
    n = X[0].shape[0]
    nn = 10#int(n/10)
    var0 = np.zeros(nn)
    for ii in np.arange(nn):
        index2 = np.arange(ii)
        index1 = np.arange(n-ii,n)
         
        xs = np.delete(X[ii],index1,0)
        x_s = np.delete(X[ii],index2,0)
         
        var0[ii] =2*(1-ii/n) * autocov(theta,xs,x_s)
     
    var0[0] = var0[0]/2.0 
    var1 = np.mean(var0)* np.eye(m)#np.outer(np.ones(m),np.ones(m))
    var_value = np.mean(var0)
     
    return var_value

# computing autocovariance for all the bags
def var_bag(theta,X,m):
    n0 = len(X)
    var0 = np.zeros(n0)
    for ii in np.arange(n0):
        xx = X[ii]
        var0[ii] = var_ts(theta,xx,m)[0,0]
    return var0


def lnd_choe(n0,n1,l,X):
    """
    choosing the landmark points
    where n0 is the number of points we choose in each bag
    n1 is the number of points in each bag
    l is the number of bags
    X is the training sample
    """
    
    m0 = X[0].shape[1]
    lnd_pts = np.zeros((n0+1,l,m0))
    for ii in np.arange(l):
        lnd_pts[0,ii] = np.mean(X[ii],0)
        loc = np.random.choice(n1,n0)
        for jj in np.arange(n0):
            lnd_pts[jj+1,ii] = X[ii][loc[jj],:]
    lnd_pts0 = lnd_pts[0]
    for ii in np.arange(1,n0+1):
        lnd_pts0 = np.vstack((lnd_pts0,lnd_pts[ii]))
    return lnd_pts0



def emp_ebeding(theta,X,Z):
    """
    computing the empirical embedding, where X is the data point in each bag,
    Y is the number of z landmark points we use
    """
       
    kk0 = sqkernel(theta,X,Z)
    mu_hat = np.mean(kk0,0)
    
    n = kk0.shape[0]
    nn = int(n/10)
    m = kk0.shape[1]
    
    autocov0 = np.zeros((n,m))
    
    for ii in np.arange(m):
        for jj in np.arange(nn):
            index2 = np.arange(jj)
            index1 = np.arange(n-jj,n)
               
            xs = np.delete(kk0[:,ii],index1,0)
            x_s = np.delete(kk0[:,ii],index2,0)
            autocov0[jj,ii] =2*(1-ii/n) * np.cov(theta,xs,x_s)[0,1]
    
    autocov0[0,:] = autocov0[0,:]/2.0
    
    autocov1 = np.diag(np.mean(autocov0,0))
    
            
    return mu_hat, autocov0, autocov1
    
    
       
# define the function to compute the corrected mean embedding and covariance for one bag        
def data_cor(theta,eta,X,Z):
    """
    X is the data points for one bag
    Z is the landmark points
    """
    m = Z.shape[0]
    
    ri = smooth_sqkernel(theta, eta, Z)
    vari = var_ts(theta,X,m)
    mu_hati,autocov_matrixi,autocov_zi = emp_ebeding(theta,X,Z)
    
    invi = np.linalg.inv(ri + vari)#+ 2*np.eye(m))
    
    mu_bari = ri.dot(invi).dot(mu_hati)
    sigma_mui = ri - ri.dot(invi).dot(ri)
    
    return mu_bari,sigma_mui        

# function to compute the plain embedding without correlation correction for all the bags
def pln_ebeding(theta,X,Z):
    n = len(X)
    m = Z.shape[0]
    
    mu_hat = np.zeros((n,m))
    
    for ii in np.arange(n):
        xx = X[ii]
        mu_hati,autocov_matrixi,autocov_zi = emp_ebeding(theta, xx, Z)
        
        mu_hat[ii,:] = mu_hati
    return mu_hat


    
def bag_cor(theta,eta,X,Z):
   
    """
    computing the correlation corrected mean and variance for every bag 
    """
    
    
    n = len(X)
    m = Z.shape[0]
    
    mu_bar = np.zeros((n,m))
    sigma_mu = list()
    
    for ii in np.arange(n):
        xx = X[ii]
        
        mu_bari,sigma_mui = data_cor(theta,eta,xx,Z)
        
        mu_bar[ii,:] = mu_bari
        sigma_mu.append(sigma_mui)
    return mu_bar, sigma_mu


# # define the function to perform newton raphson
# 
# def beta_update(obs,mu_bar,sigma_mu,beta):
#     n = len(obs)
#     m = mu_bar.shape[1]
#     
#     Yi = obs-mu_bar.dot(beta)
#     sigmai0 = beta.dot(sigma_mu).dot(beta)
#     sigma_max = 2*np.amax(sigmai0)
#     sigma_xi = np.sqrt(sigma_max-sigmai0)
#     Xi = np.random.normal(0,sigma_xi) 
#     Zi = Xi + Yi
#     sigma_hat = np.mean(Zi**2)-sigma_max
#     
#        
#     dbeta = np.zeros(m)
#     Ibeta = np.zeros((m,m))
#     for jj in np.arange(n):
#         sigmaj = beta.dot(sigma_mu[jj]).dot(beta)+sigma_hat
#         dbetaj = (obs[jj]-beta.dot(mu_bar[jj,:]))/sigmaj * mu_bar[jj,:]+(obs[jj]-beta.dot(mu_bar[jj,:]))**2/(sigmaj**2)*(sigma_mu[jj].dot(beta))-(sigma_mu[jj].dot(beta))/sigmaj
#         Ibetaj = np.outer(mu_bar[jj,:],mu_bar[jj,:])/sigmaj + 2*sigma_mu[jj].dot(np.outer(beta,beta)).dot(sigma_mu[jj])/sigmaj**2
#         dbeta = dbeta + dbetaj
#         Ibeta = Ibeta + Ibetaj
#     beta_new = beta+ 0.00005*dbeta  # np.linalg.solve(Ibeta+0.1*np.eye(m),dbeta)
#     
#     Yi_new = obs-mu_bar.dot(beta_new)
#     sigmai0_new = beta_new.dot(sigma_mu).dot(beta_new)
#     sigma_max_new = 2*np.amax(sigmai0_new)
#     sigma_xi_new = np.sqrt(sigma_max_new-sigmai0_new)
#     Xi_new = np.random.normal(0,sigma_xi_new) 
#     Zi_new = Xi_new + Yi_new
#     sigma_hat_new = np.mean(Zi_new**2)-sigma_max_new
#     
#     lkd = 0
#     for jj in np.arange(n):
#         sigmaj = beta_new.dot(sigma_mu[jj]).dot(beta_new)+sigma_hat_new
#         lkd0 = -0.5*((obs[jj]-beta_new.dot(mu_bar[jj,:]))**2/ sigmaj + np.log(sigmaj))
#         lkd = lkd + lkd0
#     return beta_new, lkd,dbeta,Ibeta
# 
# 
# def para_est(obs,X,Z,theta,eta,nn):
#     mu_bar, sigma_mu = bag_cor(theta,eta,X,Z)
#     
#     beta0 = np.linalg.inv(np.dot(mu_bar.T,mu_bar)+0.1*np.eye(mu_bar.shape[1])).dot(mu_bar.T).dot(obs)
#     
#     beta = [beta0]
#     #beta_d_norm = np.zeros(nn)
#     #beta_norm = np.zeros(nn)
#     lkd0 = np.zeros(nn)
#     for ii in np.arange(1,nn):
#         beta_new,lkd0[ii],dbeta0,Ibeta0 = beta_update(obs,mu_bar,sigma_mu,beta[ii-1])
#         beta.append(beta_new)
#         #beta_d_norm[ii] = np.linalg.norm(beta[ii]-beta[ii-1])
#         #beta_norm[ii] = np.linalg.norm(beta[ii])
#     beta_results = beta[nn-1]
#     prect = mu_bar.dot(beta_results)
#     lkd0 = np.delete(lkd0,0)
#     return beta0,beta_results,lkd0


# median heuristic
def med_heu(l,n,X):
    """
    use median heuristic to compute the kernel parameter
    l is the number of bags
    n is the number of bags we choose from l bags
    X is the training sample
    """
    
    loc_bag = np.random.choice(l,n)
    bag_median = X[loc_bag[0]]
    for ii in np.arange(1,n):
        bag_median = np.vstack((bag_median,X[loc_bag[ii]]))
    dist_bag = spd.pdist(bag_median)
    theta0 = np.mean(dist_bag)
    return(theta0)


