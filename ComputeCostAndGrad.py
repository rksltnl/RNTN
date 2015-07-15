import numpy as np
import utils
from RNTNModel import RNTNModel


class ComputeCostAndGrad:
    """
    Input: model, data
    Output: cost, grad
    """

    def __init__(self, dictionary, trees_train, trees_dev=None):

        self.dictionary = dictionary
        self.trees_train = trees_train
        self.trees_dev = trees_dev

        self.loss = 0.0
        self.dJ_dWs = None
        self.dJ_dL = None
        self.dJ_dW = None
        self.dJ_dV = None

    def compute(self, theta=None):

        # Create a new model from scratch
        model = RNTNModel(self.dictionary)
        if theta is not None:
            model.updateParamsGivenTheta(theta)

        # Initialize all parameters
        cost = 0.0
        grad = np.zeros(model.num_parameters)

        self.loss = 0.0

        self.dJ_dWs = np.zeros(model.Ws.shape)
        self.dJ_dL = np.zeros(model.L.shape)
        self.dJ_dW = np.zeros(model.W.shape)
        self.dJ_dV = np.zeros(model.V.shape)

        # Copy tree and forward prop to populate node vectors
        #   return the cost of the network
        tree_train_clone = []
        for tree in self.trees_train:
            cloned_tree = tree.clone()
            self.forwardPass(model, cloned_tree)
            tree_train_clone.append(cloned_tree)

        # Compute cost: sum of the prediction loss and add regularization terms
        cost = self.loss + self.calculateRegularizationCost(model)

        # Backprop on cloned trees
        #   return the gradient of the network params
        for tree in tree_train_clone:
            dJ_dz_prop = np.zeros(model.dim)
            self.backwardPass(model, tree, dJ_dz_prop)

        # Compute full gradient: sum of gradient matrices and \Delta J_reg terms
        grad = self.calculateTotalGradient(model)
        return cost, grad

    def forwardPass(self, model, tree):

        # Traverse the tree and populate word vectors and
        #   predictions, bottom up

        if tree.is_leaf():
            #word_index = model.word_lookup[tree.word]
            word_index = self.getWordIndex(
                model, tree.word)
            tree.word_vector = model.L[:, word_index]
        else:
            left_child = tree.subtrees[0]
            right_child = tree.subtrees[1]
            self.forwardPass(model, left_child)
            self.forwardPass(model, right_child)

            tree.word_vector = self.composition(
                model, left_child.word_vector, right_child.word_vector)

        # hit elemenwise tanh
        tree.word_vector = np.tanh(tree.word_vector)
        # make softmax prediction
        tree.prediction = utils.softmax(
            model.Ws.dot(np.append(tree.word_vector, [1])))

        # update (increment) loss
        label_vector = self.getLabelVector(model, tree.label)
        #self.loss += -1*sum(label_vector*np.log(tree.prediction))
        self.loss += -1*label_vector.dot(np.log(tree.prediction))

    def backwardPass(self, model, tree, dJ_dz_prop):

        # Update classification matrix derivative: dJ_dWs
        word_vector_with_bias = np.append(tree.word_vector, [1])
        prediction_diff = tree.prediction - self.getLabelVector(model, tree.label)

        # check if this is a matrix
        self.dJ_dWs += np.outer(prediction_diff, word_vector_with_bias)
        assert self.dJ_dWs.shape == model.Ws.shape, \
            "classification matrix dim is incorrect"

        # Compute dJ_dz
        dJ_dz_pred = model.Ws[:,:-1].T.dot(prediction_diff)*(1 - tree.word_vector**2)
        # Add dJ_dz_prop
        dJ_dz_full = dJ_dz_pred + dJ_dz_prop

        # Branch based on leaf vs non-leaf nodes
        if tree.is_leaf():
            # Leaf node update L matrix
            #word_index = model.word_lookup[tree.word]
            word_index = self.getWordIndex(model, tree.word)
            self.dJ_dL[:, word_index] += dJ_dz_full
        else:
            # None leaf node updates W, V matrices
            c_vector = np.hstack([tree.subtrees[0].word_vector, tree.subtrees[1].word_vector])

            self.dJ_dW += np.outer(dJ_dz_full, np.append(c_vector, [1]))
            assert self.dJ_dW.shape == model.W.shape,\
                "composition W dim is incorrect"

            if model.use_tensor:
                self.dJ_dV += np.tensordot(dJ_dz_full, np.outer(c_vector, c_vector), axes=0).T

            # Compute the down layer dJ_dz^1 derivative from dJ_dz^0 derivative
            dJ_dz_down = model.W[:,:-1].T.dot(dJ_dz_full)

            if model.use_tensor:
                dJ_dz_down += (model.V + np.transpose(model.V, axes=[1,0,2])).T.dot(c_vector).T.dot(dJ_dz_full)

            assert dJ_dz_down.size == model.dim*2,\
                "down gradient dim is incorrect"

            dJ_dz_down = dJ_dz_down * (1 - c_vector**2)

            dJ_dz_down_left = dJ_dz_down[:model.dim]
            dJ_dz_down_right = dJ_dz_down[model.dim:]
            assert dJ_dz_down_left.size == dJ_dz_down_right.size, \
                "down gradient left&right dim mismatch"

            self.backwardPass(model, tree.subtrees[0], dJ_dz_down_left)
            self.backwardPass(model, tree.subtrees[1], dJ_dz_down_right)


    # Helper functions
    def calculateRegularizationCost(self, model):
        # initialize with prediction loss
        reg = 0.0

        # add Ws regularization term
        reg = len(self.trees_train) * model.lambda_Ws/2 * np.linalg.norm(model.Ws)**2
        # add L regularization term
        reg += len(self.trees_train) * model.lambda_L/2 * np.linalg.norm(model.L)**2
        # add W regularization term
        reg += len(self.trees_train) * model.lambda_W/2 * np.linalg.norm(model.W)**2

        if model.use_tensor:
            reg += len(self.trees_train) * model.lambda_V/2 * np.linalg.norm(model.V)**2

        return reg

    def calculateTotalGradient(self, model):
        grad = np.zeros(model.num_parameters)

        # add regularizer gradients
        self.dJ_dWs += len(self.trees_train) * model.lambda_Ws * model.Ws
        self.dJ_dL += len(self.trees_train) * model.lambda_L * model.L
        self.dJ_dW += len(self.trees_train) * model.lambda_W * model.W

        if model.use_tensor:
            self.dJ_dV += len(self.trees_train)*model.lambda_V * model.V

            grad = utils.vectorizeParams(
                self.dJ_dWs, self.dJ_dL, self.dJ_dW, self.dJ_dV)
        else:
            grad = utils.vectorizeParams(
                self.dJ_dWs, self.dJ_dL, self.dJ_dW)

        return grad

    def composition(self, model, child1, child2):
        c_vector = np.hstack([child1, child2])
        word_vector = model.W.dot(np.append(c_vector, [1]))

        if model.use_tensor:
            word_vector += c_vector.T.dot(model.V.T).dot(c_vector)

        return word_vector

    def getLabelVector(self, model, label):
        label_vector = np.zeros(model.K)
        node_label = (int)(label)
        label_vector[node_label] = 1
        return label_vector

    def getWordIndex(self, model, word):
        # Deal with previously unseen words
        if word in model.word_lookup:
            word_index = model.word_lookup[word]
        else:
            word_index = model.word_lookup[model.UNKNOWN_WORD]

        return word_index
