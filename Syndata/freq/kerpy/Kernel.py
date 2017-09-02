from abc import abstractmethod
from numpy import eye, concatenate, zeros, shape, mean, reshape, arange, exp, outer,\
    linalg, dot, cos, sin, sqrt
from numpy.random import permutation
from numpy.lib.index_tricks import fill_diagonal
from matplotlib.pyplot import imshow,show
import numpy as np
import warnings
from sklearn import cross_validation
from tools.GenericTests import GenericTests

class Kernel(object):
    def __init__(self):
        self.rff_num=None
        self.rff_freq=None
        pass
    
    def __str__(self):
        s=""
        return s
    
    @abstractmethod
    def kernel(self, X, Y=None):
        raise NotImplementedError()
    
    @abstractmethod
    def set_width(self, width):
        if hasattr(self, 'width'):
            warnmsg="\nChanging kernel width from "+str(self.width)+" to "+str(width)
            warnings.warn(warnmsg)
            if self.rff_freq is not None:
                warnmsg="\nrff frequencies found. rescaling to width " +str(width)
                warnings.warn(warnmsg)
                self.rff_freq=self.unit_rff_freq/width
            self.width=width
        else:
            raise ValueError("Senseless: kernel has no 'width' attribute!")
    
    @abstractmethod
    def rff_generate(self,m,dim=1):
        raise NotImplementedError()
    
    @abstractmethod
    def rff_expand(self,X):
        if self.rff_freq is None:
            raise ValueError("rff_freq has not been set. use rff_generate first")
        """
        Computes the random Fourier features for the input dataset X
        for a set of frequencies in rff_freq.
        This set of frequencies has to be precomputed
        X - 2d numpy.ndarray, first set of samples:
            number of rows: number of samples
            number of columns: dimensionality
        """
        GenericTests.check_type(X, 'X',np.ndarray)
        xdotw=dot(X,(self.rff_freq).T)
        return sqrt(2./self.rff_num)*np.concatenate( ( cos(xdotw),sin(xdotw) ) , axis=1 )
        
    @abstractmethod
    def gradient(self, x, Y):
        
        # ensure this in every implementation
        assert(len(shape(x))==1)
        assert(len(shape(Y))==2)
        assert(len(x)==shape(Y)[1])
        
        raise NotImplementedError()
    
    @staticmethod
    def centering_matrix(n):
        """
        Returns the centering matrix eye(n) - 1.0 / n
        """
        return eye(n) - 1.0 / n
    
    @staticmethod
    def center_kernel_matrix(K):
        """
        Centers the kernel matrix via a centering matrix H=I-1/n and returns HKH
        """
        n = shape(K)[0]
        H = eye(n) - 1.0 / n
        return  1.0 / n * H.dot(K.dot(H))
    
    @staticmethod
    def unbiased_HSnorm_estimate_of_centred_operator(K):
        '''returns an unbiased estimate of \sum_r \lambda_r^2 where \lambda_r are the 
        eigenvalues of the centred kernel operator'''
        m = shape(K)[0]
        fill_diagonal(K,0.)
        first_term = sum(K**2)
        second_term = (sum(K)**2)/((m-1.)*(m-2.))
        third_term = -2.*sum(K.dot(K))/(m-2.)
        return (first_term+second_term+third_term)/(m*(m-3.))
    
    @abstractmethod
    def show_kernel_matrix(self,X,Y=None):
        K=self.kernel(X,Y)
        imshow(K, interpolation="nearest")
        show()
    
    
    @abstractmethod
    def ridge_regress(self,X,y,lmbda=0.01,Xtst=None,ytst=None):
        K=self.kernel(X)
        n=shape(K)[0]
        aa=linalg.solve(K+lmbda*eye(n),y)
        if Xtst is None:
            return aa
        else:
            ypre=dot(aa,self.kernel(X,Xtst))
            if ytst is None:
                return aa,ypre
            else:
                return aa,ypre,linalg.norm(ytst-ypre)**2/len(ytst)
    
    @abstractmethod
    def ridge_regress_rff(self,X,y,lmbda=0.01,Xtst=None,ytst=None):
#         if self.rff_freq is None:
#             warnings.warn("\nrff_freq has not been set!\nGenerating new random frequencies (m=100 by default)")
#             self.rff_generate(100,dim=shape(X)[1])
#             print shape(X)[1]
        phi=self.rff_expand(X)
        bb=linalg.solve(dot(phi.T,phi)+lmbda*eye(self.rff_num),dot(phi.T,y))
        if Xtst is None:
            return bb
        else:
            phitst=self.rff_expand(Xtst)
            ypre=dot(phitst,bb)
            if ytst is None:
                return bb,ypre
            else:
                return bb,ypre,mean(linalg.norm(ytst-ypre)**2)
    
    @abstractmethod
    def xvalidate( self,X,y, method = 'ridge_regress',  \
                                    lmbda_grid=(1+arange(25))/200.0,  \
                                    width_grid=exp(-13+arange(25)),  \
                                    numFolds = 10  ):
        which_method = getattr(self,method)
        n=len(X)
        kf=cross_validation.KFold(n,n_folds=numFolds)
        xvalerr=zeros((len(lmbda_grid),len(width_grid)))
        width_idx=0
        for width in width_grid:
            self.set_width(width)
            lmbda_idx=0
            for lmbda in lmbda_grid:
                fold = 0
                prederr = zeros(numFolds)
                for train_index, test_index in kf:
                    if type(X)==list:
                        #could use slicing to speed this up when X is a list
                        #currently uses sklearn cross_validation framework which returns indices as arrays
                        #so simple list comprehension below
                        X_train = [X[i] for i in train_index]
                        X_test = [X[i] for i in test_index]
                    else:
                        X_train, X_test = X[train_index], X[test_index]
                    if type(y)==list:
                        y_train = [y[i] for i in train_index]
                        y_test = [y[i] for i in test_index]
                    else:
                        y_train, y_test = y[train_index], y[test_index]
                    _,_,prederr[fold]=which_method(X_train,y_train,lmbda=lmbda,Xtst=X_test,ytst=y_test)
                    fold+=1
                xvalerr[lmbda_idx,width_idx]=mean(prederr)
                lmbda_idx+=1
            width_idx+=1
        min_idx = np.unravel_index(np.argmin(xvalerr),shape(xvalerr))
        return lmbda_grid[min_idx[0]],width_grid[min_idx[1]]
    
    @abstractmethod
    def estimateMMD(self,sample1,sample2,unbiased=False):
        """
        Compute the MMD between two samples
        """
        K11 = self.kernel(sample1)
        K22 = self.kernel(sample2)
        K12 = self.kernel(sample1,sample2)
        if unbiased:
            fill_diagonal(K11,0.0)
            fill_diagonal(K22,0.0)
            n=float(shape(K11)[0])
            m=float(shape(K22)[0])
            return sum(sum(K11))/(pow(n,2)-n) + sum(sum(K22))/(pow(m,2)-m) - 2*mean(K12[:])
        else:
            return mean(K11[:])+mean(K22[:])-2*mean(K12[:])
        
    @abstractmethod
    def estimateMMD_rff(self,sample1,sample2,unbiased=False):
#         if self.rff_freq is None:
#             warnings.warn("\nrff_freq has not been set!\nGenerating new random frequencies (m=100 by default)")
#             self.rff_generate(100,dim=shape(sample1)[1])
        phi1=self.rff_expand(sample1)
        phi2=self.rff_expand(sample2)
        featuremean1=mean(phi1,axis=0)
        featuremean2=mean(phi2,axis=0)
        if unbiased:
            nx=shape(phi1)[0]
            ny=shape(phi2)[0]
            first_term=nx/(nx-1.0)*( dot(featuremean1,featuremean1)   \
                                        -mean(linalg.norm(phi1,axis=1)**2)/nx )
            second_term=ny/(ny-1.0)*( dot(featuremean2,featuremean2)   \
                                        -mean(linalg.norm(phi2,axis=1)**2)/ny )
            third_term=-2*dot(featuremean1,featuremean2)
            return first_term+second_term+third_term
        else:
            return linalg.norm(featuremean1-featuremean2)**2
