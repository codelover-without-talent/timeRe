import numpy as np
from kerpy.MaternKernel import MaternKernel
from kerpy.GaussianKernel import GaussianKernel

class UnitTests():
    @staticmethod
    def UnitTestBagKernel(which_bag_kernel):
            num_bagsX = 20
            num_bagsY = 30
            shift = 2.0
            dim = 3
            bagsize = 50
            qvar = 0.6
            baglistx = list()
            baglisty = list()
            for _ in range(num_bagsX):
                muX = np.sqrt(qvar) * np.random.randn(1, dim)
                baglistx.append(muX + np.sqrt(1 - qvar) * np.random.randn(bagsize, dim))
            for _ in range(num_bagsY):
                muY = np.sqrt(qvar) * np.random.randn(1, dim)
                muY[:, 0] = muY[:, 0] + shift
                baglisty.append(muY + np.sqrt(1 - qvar) * np.random.randn(bagsize, dim))
            data_kernel = GaussianKernel(1.0)
            bag_kernel = which_bag_kernel(data_kernel)
            bag_kernel.show_kernel_matrix(baglistx + baglisty)
            bag_kernel.rff_generate(12,10,dim=3)
            bagmmd = bag_kernel.estimateMMD_rff(baglistx, baglisty)
            print '...successfully computed mmd on bags: ', bagmmd
            response_y=np.random.randn(num_bagsX)
            b=bag_kernel.xvalidate(baglistx,response_y,'ridge_regress_rff')
            print b
            print '...successfully ran ridge regression on bags.'
            print 'unit test ran for ', bag_kernel.__str__()
