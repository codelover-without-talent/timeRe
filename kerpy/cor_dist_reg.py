from __future__ import division
import numpy as np
import scipy.stats as sps
import scipy.spatial.distance as spd

from kerpy.Kernel import Kernel
from kerpy.BagKernel import BagKernel
from kerpy.GaussianKernel import GaussianKernel
from kerpy.GaussianBagKernel import GaussianBagKernel



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


def cor_sample_gen(l,n,rho,sigma,sigma_label):
    samples = list()
    samples_statn = list()
    entro = 0.5 * np.log(2*np.pi*np.e*sigma**2)
    entro_label = np.zeros(l)
    
    
    for ii in np.arange(l):
        sample = cor_sample(n,rho,sigma[ii]).reshape((n,1))
        sample_statn = np.random.normal(0,sigma[ii],n).reshape((n,1))
        entro_label[ii] = entro[ii] + np.random.normal(0,sigma_label,1)
        
        samples.append(sample)
        samples_statn.append(sample_statn)
    return samples,samples_statn,entro,entro_label



# define the data kernel as well as bag kernel (note here for simplicity, we use the same kernel with different parameters to do regression)
def gskernel(theta,X, Y=None):
    """
    Computes r, the data kernel
    
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
        
    K = np.exp(-1* sq_dists/(2 * theta**2))
    return K

# the kernel parameter used in define the prior for mean embedding
#prior_theta = 4.0


# the kernel parameter used at data level
#data_theta = prior_theta/8.0

# define auto-covariance matrix for any two vector
def hsic_ts(theta,X,Y):
    len_series = X.shape[0]
    center_matrix = np.eye(len_series)-(1.0/len_series)*np.outer(np.ones(len_series),np.ones(len_series))
    Ks = gskernel(theta,X)
    Kss = gskernel(theta,Y)
    covxy = 1.0/ len_series**2 * np.trace(Ks.dot(center_matrix).dot(Kss).dot(center_matrix))
    
    return np.sqrt(covxy)


# define function to compute the variance of a certain time series X
def var_ts(theta,X):
    nn = X.shape[0]
    n0 = int(nn/2)
    gammas = np.zeros(n0)
    coeff = np.zeros(n0)
    
    for ii in np.arange(n0):
        s_index = np.arange(ii)
        ss_index = np.arange(n0-ii,n0)
        
        xs = np.delete(X,s_index,0)
        xss = np.delete(X,ss_index,0)
        gammas[ii] = hsic_ts(theta,xs,xss)
        coeff[ii] = 2 * (1-ii/n0)
    coeff[0] = 1
    
    gamma_squared = coeff.dot(gammas)/n0
   
    return gamma_squared

# var0 = var_ts(data_theta,samp0[1])
# print var0

def lnd_choe(n0,X):
    """
    choosing the landmark points
    where n0 is the number of points we choose in each bag
    X is the training sample
    """

    n1 = X[0].shape[0]
    m0 = X[0].shape[1]
    l = len(X)
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
#lnd_bag = 3
#ldmk0 = lnd_choe(lnd_bag,samp0)
# print ldmk0.shape[0]

def emp_embedding(theta,X,Z):
    """
    the function is used to compute the empirical mean embedding given the time series X and landmark point Z
    """
    embedding_kernel = gskernel(theta,X,Z)
    mean_embedding = np.mean(embedding_kernel,0)
    
    return mean_embedding



# define the function to compute the empirical mean embedding for all the bags
def emp_bag(theta,X,Z):
    l1 = len(X)
    l2 = Z.shape[0]
    
    embed_matrix = np.zeros((l1,l2))
    
    for ii in np.arange(l1):
        sampi = X[ii]
        
        empirical_embedding_i = emp_embedding(theta,sampi,Z)
        embed_matrix[ii,:] = empirical_embedding_i
        
    return embed_matrix 


def shrin_embedding(data_theta,prior_theta,X,Z,m0):
    """
    the function is to compute the shrinkage mean embedding
    data_theta is the parameter used in kernel for data level 
    prior_theta is the parameter used to construct the prior for mean embeddings
    X is the given time sereis
    Z is the landmark point
    m0 is the initial mean we wish shrink to
    """
    mean_embedding = emp_embedding(data_theta,X,Z)
    #print mean_embedding[:10]
    ri = gskernel(prior_theta,Z)
    #print ri
    
    li = Z.shape[0]
    sigmai = var_ts(data_theta,X)*np.eye(li)
    
    crct_embe = ri.dot(np.linalg.solve(ri+sigmai,mean_embedding-m0))+m0
    #print crct_embe[:10]
    crct_var = ri-ri.dot(np.linalg.solve(ri+sigmai,ri))
    #print crct_var
    return crct_embe,crct_var


# define the function to compute the shrinkage mean embedding for all the bags
def shrin_bag(data_theta,prior_theta,X,Z,m0):
    l1 = len(X)
    l2 = Z.shape[0]
    
    embeding_matrix = np.zeros((l1,l2))
    var_bag = list()
    
    for ii in np.arange(l1):
        sampi = X[ii]
        
        cor_emb,cor_var = shrin_embedding(data_theta,prior_theta,sampi,Z,m0)
        
        embeding_matrix[ii,:] = cor_emb
        var_bag.append(cor_var)
    return embeding_matrix,var_bag
# 
#shrin_embed_matrix0,shrin_var_bag0 = shrin_bag(data_theta,prior_theta,samp0,ldmk0,m0)
#shrin_embed_matrix1,shrin_var_bag1 = shrin_bag(data_theta,prior_theta,samp1,ldmk0,m1)
#print shrin_embed_matrix0



# define the function to perform regression
def dist_reg(X,Y,lmbda,xtst=None,ytst=None):
    l0 = X.shape[1]
    matrix1 = X.T.dot(X)+lmbda*np.eye(l0)
    matrix2 = X.T.dot(Y)
    
    beta = np.linalg.solve(matrix1,matrix2)
    Y_hat = X.dot(beta)
    rmse = np.sqrt(np.mean((Y-Y_hat)**2)) 
    if xtst is None:
        return beta,Y_hat,rmse
    else:
        ytst_hat = xtst.dot(beta)
        rmse = np.sqrt(np.mean((ytst-ytst_hat)**2))
        return beta,ytst_hat,rmse
    
#lmbda0 = 0.001


# # define the function to perform newton raphson 
def beta_update(obs,mu_bar,sigma_mu,beta,sigma_ini):
    n = len(obs)
    m = mu_bar.shape[1]
    # number of iterations for estimating sigma
    n0 = 100
    
    sigmai0 = beta.dot(sigma_mu).dot(beta)
    sigma_estimate = [sigma_ini]
    for ii in np.arange(n0):
        dsigma = 0
        Isigma = 0
        sigma0 = sigma_estimate[ii]
    
        for jj in np.arange(n):
            dsigmaj = 0.5 * ((obs[jj]-mu_bar[jj,:].dot(beta))**2/(sigmai0[jj]+sigma0)**2-1/(sigmai0[jj]+sigma0))
            Isigmaj = 0.5*(1/(sigmai0[jj]+sigma0)**2- 2*(obs[jj]-mu_bar[jj,:].dot(beta))**2/(sigmai0[jj]+sigma0)**3)
            dsigma = dsigma+dsigmaj
            Isigma = Isigma + Isigmaj
        sigma1 = sigma0+0.01*dsigma#/Isigma
        sigma_estimate.append(sigma1)    
    sigma_hat = sigma_estimate[n0-1]
    print sigma_hat 
#     Yi = obs-mu_bar.dot(beta)
#     #print Yi
#     sigmai0 = beta.dot(sigma_mu).dot(beta)
#     #print sigmai0
#     sigma_max = np.amax(sigmai0)
#     #print sigma_max
#     sigma_xi = np.sqrt(sigma_max-sigmai0)
#     #print sigma_xi
#     Xi = np.random.normal(0,sigma_xi)
#     #print Xi 
#     Zi = Xi + Yi
#     #print Zi
#     sigma_hat = np.mean(Zi**2)-sigma_max
#     print sigma_hat 
        
    dbeta = np.zeros(m)
    Ibeta = np.zeros((m,m))
    for jj in np.arange(n):
        sigmaj = sigmai0[jj]+sigma_hat
        dbetaj = (obs[jj]-mu_bar[jj,:].dot(beta))/sigmaj * mu_bar[jj,:]+(obs[jj]-mu_bar[jj,:].dot(beta))**2/(sigmaj**2)*(sigma_mu[jj].dot(beta))-(sigma_mu[jj].dot(beta))/sigmaj
        Ibetaj = np.outer(mu_bar[jj,:],mu_bar[jj,:])/sigmaj + 2*sigma_mu[jj].dot(np.outer(beta,beta)).dot(sigma_mu[jj])/sigmaj**2
        dbeta = dbeta + dbetaj
        Ibeta = Ibeta + Ibetaj
    print Ibeta
    beta_new = beta + 0.1*dbeta   #np.linalg.solve(Ibeta,dbeta) 
     
#     Yi_new = obs-mu_bar.dot(beta_new)
#     sigmai0_new = beta_new.dot(sigma_mu).dot(beta_new)
#     sigma_max_new = 2*np.amax(sigmai0_new)
#     sigma_xi_new = np.sqrt(sigma_max_new-sigmai0_new)
#     Xi_new = np.random.normal(0,sigma_xi_new) 
#     Zi_new = Xi_new + Yi_new
#     sigma_hat_new = np.mean(Zi_new**2)-sigma_max_new
#     print sigma_hat_new
      
    lkd = 0
    for jj in np.arange(n):
        sigmaj_new = sigmai0[jj]+sigma_hat
        lkd0 = -0.5*((obs[jj]-beta_new.dot(mu_bar[jj,:]))**2/ sigmaj_new + np.log(sigmaj_new))
        lkd = lkd + lkd0
    
    return beta_new, sigma_hat,lkd






def tr_tst_split(X,Y,NumFolds):
    """
    define function to split data into K-folds
    X is the independent variable
    Y is the label
    """
    
    n = len(Y)
    
    data_index = np.arange(n)
    
    np.random.shuffle(data_index)
    
    split_index = np.split(data_index,NumFolds)
    
    train_X = list()
    test_X  = list()
    train_Y = list()
    test_Y = list()
    
    for ii in np.arange(NumFolds):
        index = list(split_index)
        test_index = index[ii]
        del index[ii]
        train_index = np.concatenate(index)
            
        train_x0 = list(X[jj] for jj in train_index)
        train_y0 = list(Y[jj] for jj in train_index)
        
        test_x0 = list(X[jj] for jj in test_index)
        test_y0 = list(Y[jj] for jj in test_index)
      
        train_X.append(train_x0)
        train_Y.append(train_y0)
        test_X.append(test_x0)
        test_Y.append(test_y0)
    return train_X,train_Y,test_X,test_Y


# define the cross validation function for empirical mean embedding regression
def emp_statn_xval(par,lnd_mk,X,Y,NumFolds):
    train_x,train_y,test_x,test_y = tr_tst_split(X,Y,NumFolds)
    
    prior_theta = par[0]
    data_theta = par[1]
    lmbda = par[2]
    
    rmse_vec = np.zeros(NumFolds)
    
    for ii in np.arange(NumFolds):
        train_xx = train_x[ii]
        train_yy = train_y[ii]
        test_xx = test_x[ii]
        test_yy = test_y[ii]
        
        emp_embed_matrix_tr = emp_bag(data_theta, train_xx, lnd_mk)
        emp_embed_matrix_tt = emp_bag(data_theta,test_xx,lnd_mk)
        _,_,rmse =dist_reg(emp_embed_matrix_tr,train_yy,lmbda,emp_embed_matrix_tt,test_yy)
        rmse_vec[ii] = rmse
    aver_rmse = np.mean(rmse_vec)
    rmse_dict = {aver_rmse:par}
    return rmse_dict

def shrin_xval(par,lnd_mk,X,Y,NumFolds):
    train_x,train_y,test_x,test_y = tr_tst_split(X,Y,NumFolds)
    
    prior_theta = par[0]
    data_theta = par[1]
    lmbda = par[2]
    
    rmse_vec = np.zeros(NumFolds)
    
    for ii in np.arange(NumFolds):
        train_xx = train_x[ii]
        train_yy = train_y[ii]
        test_xx = test_x[ii]
        test_yy = test_y[ii]
        
        #emp_embed_matrix_tr = emp_bag(data_theta, train_xx, lnd_mk)
        #emp_embed_matrix_tt = emp_bag(data_theta,test_xx,lnd_mk)
        m0_tr = 0#np.mean(emp_embed_matrix_tr,0)
        m0_tt = 0#np.mean(emp_embed_matrix_tt,0)
        shrin_embed_matrix_tr,shrin_var_bag_tr = shrin_bag(data_theta,prior_theta,train_xx, lnd_mk,m0_tr)
        shrin_embed_matrix_tt,shrin_var_bag_tt = shrin_bag(data_theta,prior_theta,test_xx,lnd_mk,m0_tt)
        _,_,rmse =dist_reg(shrin_embed_matrix_tr,train_yy,lmbda,shrin_embed_matrix_tt,test_yy)
        rmse_vec[ii] = rmse
    aver_rmse = np.mean(rmse_vec)
    rmse_dict = {aver_rmse:par}
    return rmse_dict







def emp_gaussian_xval(par,lnd_mk,X,Y,NumFolds):
    train_x,train_y,test_x,test_y = tr_tst_split(X,Y,NumFolds)
    
    prior_theta = par[0]
    data_theta = par[1]
    bag_theta = 20.085
    lmbda = par[2]
    
    rmse_vec = np.zeros(NumFolds)
    
    kernel = GaussianKernel(float(bag_theta))
    for ii in np.arange(NumFolds):
        train_xx = train_x[ii]
        train_yy = train_y[ii]
        test_xx = test_x[ii]
        test_yy = test_y[ii]
        
        
        emp_embed_matrix_tr = emp_bag(data_theta, train_xx, lnd_mk)
        emp_embed_matrix_tt = emp_bag(data_theta,test_xx,lnd_mk)
        _,_,rmse =kernel.ridge_regress(emp_embed_matrix_tr,train_yy,lmbda,emp_embed_matrix_tt,test_yy)
        rmse_vec[ii] = rmse
    aver_rmse = np.mean(rmse_vec)
    rmse_dict = {aver_rmse:par}
    return rmse_dict

def shrin_gaussian_xval(par,lnd_mk,X,Y,NumFolds):
    train_x,train_y,test_x,test_y = tr_tst_split(X,Y,NumFolds)
    
    prior_theta = par[0]
    data_theta = par[1]
    bag_theta = 2.718
    lmbda = par[2]
    
    rmse_vec = np.zeros(NumFolds)
    kernel = GaussianKernel(float(bag_theta))
    for ii in np.arange(NumFolds):
        train_xx = train_x[ii]
        train_yy = train_y[ii]
        test_xx = test_x[ii]
        test_yy = test_y[ii]
        
        #emp_embed_matrix_tr = emp_bag(data_theta, train_xx, lnd_mk)
        #emp_embed_matrix_tt = emp_bag(data_theta,test_xx,lnd_mk)
        m0_tr = 0#np.mean(emp_embed_matrix_tr,0)
        m0_tt = 0#np.mean(emp_embed_matrix_tt,0)
        shrin_embed_matrix_tr,shrin_var_bag_tr = shrin_bag(data_theta,prior_theta,train_xx, lnd_mk,m0_tr)
        shrin_embed_matrix_tt,shrin_var_bag_tt = shrin_bag(data_theta,prior_theta,test_xx,lnd_mk,m0_tt)
        _,_,rmse =kernel.ridge_regress(shrin_embed_matrix_tr,train_yy,lmbda,shrin_embed_matrix_tt,test_yy)
        rmse_vec[ii] = rmse
    aver_rmse = np.mean(rmse_vec)
    rmse_dict = {aver_rmse:par}
    return rmse_dict
