import numpy as np
from ComputeCostAndGrad import ComputeCostAndGrad
from ComputeCostAndGradMiniBatch import ComputeCostAndGradMiniBatch
import copy
from RNTNModel import RNTNModel


def checkGradient_MiniBatch(dictionary, trees):

    model = RNTNModel(dictionary)
    theta_init = model.getTheta()

    # compute analytical gradient
    costObj = ComputeCostAndGradMiniBatch()
    cost, grad = costObj.compute(theta_init, dictionary, trees)

    eps = 1E-4
    numgrad = np.zeros(grad.shape)

    # compute numerical gradient
    for i in range(model.num_parameters):
        if i % 10 == 0:
            print '%d/%d' % (i, model.num_parameters)

        indicator = np.zeros(model.num_parameters)
        indicator[i] = 1

        theta_plus = theta_init + eps*indicator
        cost_plus, grad_plus = costObj.compute(theta_plus, dictionary, trees)

        theta_minus = theta_init - eps*indicator
        cost_minus, grad_minus = costObj.compute(theta_minus, dictionary, trees)

        numgrad[i] = (cost_plus - cost_minus)/(2*eps)

    print 'analytical gradient: ', grad
    print 'numerical gradient: ', numgrad

    normdiff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print 'Norm difference: ', normdiff
    return normdiff

def checkGradientClean(dictionary, trees):

    # Code adopted from UFLDL gradientChecker
    # http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization

    model = RNTNModel(dictionary)
    theta_init = model.getTheta()

    # compute analytical gradient
    costObj = ComputeCostAndGrad(dictionary, trees)
    cost, grad = costObj.compute(theta_init)

    eps = 1E-4
    numgrad = np.zeros(grad.shape)

    # compute numerical gradient
    for i in range(model.num_parameters):
        if i % 10 == 0:
            print '%d/%d' % (i, model.num_parameters)

        indicator = np.zeros(model.num_parameters)
        indicator[i] = 1

        theta_plus = theta_init + eps*indicator
        cost_plus, grad_plus = costObj.compute(theta_plus)

        theta_minus = theta_init - eps*indicator
        cost_minus, grad_minus = costObj.compute(theta_minus)

        numgrad[i] = (cost_plus - cost_minus)/(2*eps)

    print 'analytical gradient: ', grad
    print 'numerical gradient: ', numgrad

    normdiff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print 'Norm difference: ', normdiff
    return normdiff


def checkGradient(model, trees):

    # Code adopted from UFLDL gradientChecker
    # http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization

    computer = ComputeCostAndGrad(model, trees)
    cost, grad = computer.compute()

    eps = 1E-4
    numgrad = np.zeros(grad.shape)

    theta = model.getTheta()

    for i in range(model.num_parameters):
        if i % 10 == 0:
            print '%d/%d' % (i, model.num_parameters)

        indicator = np.zeros(model.num_parameters)
        indicator[i] = 1

        theta_plus = theta + eps*indicator
        model_plus = copy.deepcopy(model)
        model_plus.updateParamsGivenTheta(theta_plus)
        computer_plus = ComputeCostAndGrad(model_plus, trees)
        cost_plus, grad_plus = computer_plus.compute()

        theta_minus = theta - eps*indicator
        model_minus = copy.deepcopy(model)
        model_minus.updateParamsGivenTheta(theta_minus)
        computer_minus = ComputeCostAndGrad(model_minus, trees)
        cost_minus, grad_minus = computer_minus.compute()

        numgrad[i] = (cost_plus - cost_minus)/(2*eps)

    print 'analytical gradient: ', grad
    print 'numerical gradient: ', numgrad

    print 'Norm difference: '
    normdiff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    return normdiff

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / sum(e)

def vectorizeParams(*args):
    vect = np.array([])
    for matrix in args:
        vect = np.hstack([vect, matrix.ravel()])
    return vect

def constructCompactDictionary(trees):
    # Take a tree lists and union all words
    #  note: have a "*UNK*" to deal with previously unseen words
    dictionary = set()
    dictionary = dictionary.union(['*UNK*'])
    for tree in trees:
        dictionary = dictionary.union(tree.word_yield().split(' '))
    return dictionary

def constructDictionary(*args):
    # Take a list of tree lists and union all words
    dictionary = set()
    for tree_split in args:
        for trees in tree_split:
            for tree in trees:
                dictionary = dictionary.union(tree.word_yield().split(' '))
    return dictionary

'''
def simpleQuadraticFunction(x):
    cost = x[0]**2 + 3*x[0]*x[1]
    grad = np.zeros(2)
    grad[0] = 2*x[0] + 3*x[1]
    grad[1] = 3*x[0]
    return cost, grad

def checkQuadraticGradient():
    model = np.array([4,10])
    [cost, grad] = simpleQuadraticFunction(model)

    eps = 1E-4
    numgrad = np.zeros(model.shape)

    model_perturbed = model

    for i in range(2):
        indicator = np.zeros(2)
        indicator[i] = 1;

        model_perturbed = model + eps*indicator
        cost_plus, grad_plus = simpleQuadraticFunction(model_perturbed)

        model_perturbed = model - eps*indicator
        cost_minus, grad_minus = simpleQuadraticFunction(model_perturbed)

        numgrad[i] = (cost_plus - cost_minus)/(2*eps)

    print grad
    print numgrad

    print np.linalg.norm(grad - numgrad) / np.linalg.norm(grad + numgrad)
'''