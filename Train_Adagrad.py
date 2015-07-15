from ComputeCostAndGradMiniBatch import ComputeCostAndGradMiniBatch
from RNTNModel import RNTNModel
import random
import numpy as np
from Test import Test
import pickle
import time
import sys


class Train_Adagrad:

    def __init__(self, dictionary, X_train, X_dev=None, X_test=None):
        self.X_train = X_train
        self.X_dev = X_dev
        self.X_test = X_test
        self.dictionary = dictionary
        self.costObj = ComputeCostAndGradMiniBatch()
        dumb_model = RNTNModel(dictionary)
        self.theta_init = dumb_model.getTheta()
        self.num_data = len(X_train)
        self.num_parameters = dumb_model.num_parameters

        # SGD params
        self.batch_size = dumb_model.batch_size
        self.num_batches = self.num_data / self.batch_size
        self.max_epochs = dumb_model.max_epochs
        self.learning_rate = dumb_model.learning_rate
        self.fudge = 1E-3
        self.epoch_save_freq = 5  # save every 5 epochs

    def costWrapper(self, theta, X_train_mbatch):
        cost, grad = self.costObj.compute(
            theta, self.dictionary, X_train_mbatch, self.X_dev)
        return cost, grad

    def train(self):
        print "[INFO] Training .."
        grad_history = np.zeros(self.num_parameters)
        theta = self.theta_init
        # Loop over epochs
        for epochid in range(self.max_epochs):
            # create a shuffled copy of the data
            X_shuffled = random.sample(self.X_train, self.num_data)
            # reset grad history per each epoch
            grad_history = np.zeros(self.num_parameters)

            # Loop over batches
            for batch_id in range(self.num_batches):
                start_i = batch_id * self.batch_size
                end_i = (batch_id+1) * self.batch_size
                if end_i + self.batch_size > self.num_data:
                    end_i = self.num_data

                X_batch = X_shuffled[start_i:end_i]
                theta, grad_history = self.trainOneBatch(
                    theta, X_batch, grad_history, batch_id)

            print "Finished epoch %d." % epochid

            # Save the model at every 5 epochs
            if epochid % self.epoch_save_freq == 0:
                filename = "optResult-RNTN-" + \
                           time.strftime("%Y%m%d-%H%M%S") + "-epoch-" + str(epochid)
                with open(filename, 'wb') as output:
                    pickle.dump(theta, output, -1)

            # Evaluate on train, test set
            testObj_train = Test(self.dictionary, self.X_train)
            tree_accuracy_train, root_accuracy_train = testObj_train.test(theta)
            print "[Train accuracy] tree: %.2f, root: %.2f" %\
                  (tree_accuracy_train, root_accuracy_train)

            # Test on test data
            testObj_test = Test(self.dictionary, self.X_test)
            tree_accuracy_test, root_accuracy_test = testObj_test.test(theta)
            print "[Test accuracy] tree: %.2f, root: %.2f" %\
                  (tree_accuracy_test, root_accuracy_test)
            sys.stdout.flush()

        return theta

    def trainOneBatch(self, theta, X, grad_history, batch_id):

        cost, grad = self.costWrapper(theta, X)
        if batch_id % 30 == 0:
            print '%d/%d' % (batch_id, self.num_batches),
            print 'batch cost: ', cost
            sys.stdout.flush()

        grad_history_out = grad_history + grad**2
        theta_out = theta - self.learning_rate * \
                            grad / (np.sqrt(grad_history_out) + self.fudge)

        return theta_out, grad_history_out