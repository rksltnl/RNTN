from ComputeCostAndGradMiniBatch import ComputeCostAndGradMiniBatch
from RNTNModel import RNTNModel
import numpy as np


class Test:

    def __init__(self, dictionary, X):
        self.costObj = ComputeCostAndGradMiniBatch()
        self.model = RNTNModel(dictionary)
        self.trees = X

        self.num_correct = 0.0
        self.num_wrong = 0.0

        self.num_correct_root = 0.0
        self.num_wrong_root = 0.0

    def test(self, theta):
        self.model.updateParamsGivenTheta(theta)

        tree_clone = []
        for tree in self.trees:
            cloned_tree = tree.clone()
            self.costObj.forwardPass(self.model, cloned_tree)
            tree_clone.append(cloned_tree)

        # Traverse the tree and compare with labels
        for tree in tree_clone:
            self.evaluate_allnode(tree)
            self.evaluate_rootnode(tree)

        accuracy_allnode = self.num_correct/(self.num_correct + self.num_wrong)*100
        accuracy_rootnode = self.num_correct_root/(self.num_correct_root + self.num_wrong_root)*100
        return accuracy_allnode, accuracy_rootnode

    def evaluate_allnode(self, tree):
        if not tree.is_leaf():
            left_child = tree.subtrees[0]
            right_child = tree.subtrees[1]
            self.evaluate_allnode(left_child)
            self.evaluate_allnode(right_child)

        if int(tree.label) == np.argmax(tree.prediction):
            self.num_correct += 1
        else:
            self.num_wrong += 1

    def evaluate_rootnode(self, tree):
        if int(tree.label) == np.argmax(tree.prediction):
            self.num_correct_root += 1
        else:
            self.num_wrong_root += 1