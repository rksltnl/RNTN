from scipy.optimize import minimize
from ComputeCostAndGrad import ComputeCostAndGrad
from RNTNModel import RNTNModel
import sys


class Train_LBFGS:

    def __init__(self, dictionary, X):
        self.costObj = ComputeCostAndGrad(dictionary, X)
        dumb_model = RNTNModel(dictionary)
        self.theta_init = dumb_model.getTheta()

    def costWrapper(self, theta):
        cost, grad = self.costObj.compute(theta)
        print 'full batch cost: ', cost
        sys.stdout.flush()
        return cost, grad

    def train(self):
        print "[INFO] Training .."
        #res = minimize(self.costWrapper, self.theta_init,
        #               method='L-BFGS-B', jac=True, options={'maxiter':4, 'disp':True})

        res = minimize(self.costWrapper, self.theta_init,
                       method='L-BFGS-B', jac=True, options={'maxiter':200, 'disp':True})
        return res