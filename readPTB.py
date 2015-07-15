# import Jonathan K's Berkeley parser analyzer
from ptb import *

def parser(filename):
    trees = read_trees(filename)
    return trees