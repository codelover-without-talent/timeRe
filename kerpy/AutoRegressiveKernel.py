import numpy as np
from numpy import infty
from kerpy.Kernel import Kernel
from tools.GenericTests import GenericTests



class AutoRegressiveKernel(Kernel):
    def __init__(self, p=1, alpha=0.5, sigma=1.0):
        Kernel.__init__(self)       
        self.p = p
        self.alpha = alpha
        self.sigma = sigma
    
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "p="+ str(self.p)
        s += "alpha="+ str(self.alpha)
        s += ", " + Kernel.__str__(self)
        s += "]"
        return s
    
    def kernel(self, X, Y=None):
        
        GenericTests.check_type(X,'X',np.ndarray,2)
        if Y is None:
            Y=X
               
        nX=np.shape(X)[0]
        nY=np.shape(Y)[0]
        K=np.zeros((nX,nY))
        ii=0        
        for x in X:
            jj=0
            for y in Y:
                Ax,Bx=self.formVARmatrices(x)
                degx=np.shape(Bx)[1]
                Ay,By=self.formVARmatrices(y)
                degy=np.shape(By)[1]
                deltaMat = np.diag(np.concatenate((0.5*np.ones(degx)/degx,0.5*np.ones(degy)/degy)))
                A=np.concatenate((Ax,Ay),axis=1)
                B=np.concatenate((Bx,By),axis=1)
                Adel = A.dot(deltaMat)
                AdelAT = Adel.dot(A.T) 
                foo=np.linalg.solve(AdelAT+np.eye(self.p),Adel)
                precomputedMat=deltaMat-(deltaMat.dot(A.T)).dot(foo) 
                _,first_term = np.linalg.slogdet(AdelAT+np.eye(self.p))
                second_term = (B.dot(precomputedMat)).dot(B.T)+1.
                K[ii,jj]+= -(1-self.alpha)*first_term-self.alpha*second_term
                jj+=1
            ii+=1
        return np.exp(-0.5*K/(self.sigma**2.))
    
    
    def formVARmatrices(self,x):
        lenx=self.TimeSeriesLength(x)
        Ax=np.zeros((self.p,lenx-self.p))
        for ii in range(self.p):
            Ax[ii]=x[ii:(lenx-self.p+ii)]
        Bx=np.reshape(x[self.p:lenx],(1,lenx-self.p))
        return Ax,Bx
    
    def TimeSeriesLength(self,x):
        appended=np.flatnonzero(x==-infty)
        if appended.size:
            return appended[0]
        else:
            return len(x)
            
    
    def gradient(self, x, Y):
        raise NotImplementedError()
    
if __name__ == '__main__':
    kernel = AutoRegressiveKernel(p=3)
    nX=20
    nY=40
    maxlen=50
    X=np.random.randn(nX,maxlen)
    terminateX=np.random.randint(15,maxlen,nX)
    for ii in range(nX):
        X[ii,terminateX[ii]:]=-infty        
    Y=np.random.randn(nY,maxlen)
    terminateY=np.random.randint(15,maxlen,nY)
    for jj in range(nY):
        Y[jj,terminateY[jj]:]=-infty
    kernel.show_kernel_matrix(X)
    
