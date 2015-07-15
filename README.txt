< RNTN implementations with 1) SGD, and 2) LBFGS >

// author: Hyun Oh Song (hsong@cs.stanford.edu)
//           uses the Jonathan's Berkeley parser code (https://code.google.com/p/berkeley-parser-analyser/source/browse/archival_versions/emnlp2012/ptb.py) to read in PTB trees.


< Instructions > 

A. To run with provided data, type,

$python main.py

B. To run with custom data, type,

$python main.py --trainPath <path to training data> --testPath <path to test data>

C. To run gradient check with provided subset of data, type,

$python main.py --checkGradient True

// Note, this is my gradient check implementation of the tutorial at http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization, the code outputs the norm difference.)

D. To run with stdout copied to a log file, type,

$python main.py | tee <filename.txt>