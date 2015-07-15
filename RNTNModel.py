import numpy as np
import utils


class RNTNModel:

    def __init__(self, dictionary):

        # use tensor?
        self.use_tensor = True

        self.dim = 25  # word vector dimension
        self.K = 5     # num classes

        # regularization constants
        self.lambda_Ws = 0.0001
        self.lambda_L = 0.0001
        self.lambda_W = 0.001
        self.lambda_V = 0.001

        # self.lambda_Ws = 1000000
        # self.lambda_L = 1000000
        # self.lambda_W = 1000000
        # self.lambda_V = 1000000

        # Params for SGD
        self.learning_rate = 0.01
        self.max_epochs = 200
        self.batch_size = 27

        # initialize parameters uniform(-r ~ r)
        self.r = 0.0001

        # constant for previously unknown word
        self.UNKNOWN_WORD = '*UNK*'

        # Composition matrices V and W
        # self.V = np.random.uniform(-self.r, self.r,
        #                            size=(2*self.dim, 2*self.dim, self.dim))
        # self.W = np.random.uniform(-self.r, self.r,
        #                            size=(self.dim, 2*self.dim+1))
        # # Classification matrix
        # self.Ws = np.random.uniform(-self.r, self.r,
        #                            size=(self.K, self.dim+1))
        # # Word vector matrix
        # self.L = np.random.uniform(-self.r, self.r,
        #                            size=(self.dim, len(dictionary)))

        # Classification matrix (init bias column to zero)
        self.Ws = np.zeros((self.K, self.dim+1))
        range = 1.0 / np.sqrt(self.dim)
        self.Ws[:, :-1] = np.random.uniform(-1*range, range,
                                            size=(self.K, self.dim))

        # Word vector matrix
        self.L = np.random.randn(self.dim, len(dictionary))

        # Composition matrix W
        self.W = np.zeros((self.dim, 2*self.dim+1))
        range = 1.0 / (np.sqrt(self.dim) * 2.0)
        self.W[:, :-1] = np.random.uniform(-1*range,
                                           range, size=(self.dim, 2*self.dim))
        # Socher adds identity matrix here. Why?
        self.W[:, :self.dim] += np.identity(self.dim)
        self.W[:, self.dim:-1] += np.identity(self.dim)

        # Composition matrix V
        range = 1.0 / (4 * self.dim)
        self.V = np.random.uniform(-1*range, range,
                                   size=(2*self.dim, 2*self.dim, self.dim))

        # Keep total number of parameters for checks
        self.num_parameters = \
            self.W.size + self.Ws.size + self.L.size
        if self.use_tensor:
            self.num_parameters += self.V.size

        # Hash table of (string) word -> (int) index in L
        self.word_lookup = dict()
        for i, word in enumerate(dictionary):
            self.word_lookup[word] = i

    # return vectorized params
    def getTheta(self):
        if self.use_tensor:
            return utils.vectorizeParams(
                self.Ws, self.L, self.W, self.V)
        else:
            return utils.vectorizeParams(
                self.Ws, self.L, self.W)

    # update parameters from theta
    def updateParamsGivenTheta(self, theta):
        assert theta.size == self.num_parameters, \
            "[Error] input theta dim doesn't match the dimension."

        bound1 = self.Ws.size
        self.Ws = theta[:bound1].reshape(self.Ws.shape)
        bound2 = bound1 + self.L.size
        self.L = theta[bound1:bound2].reshape(self.L.shape)
        bound3 = bound2 + self.W.size
        self.W = theta[bound2:bound3].reshape(self.W.shape)

        if self.use_tensor:
            self.V = theta[bound3:].reshape(self.V.shape)


