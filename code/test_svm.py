# Test SVM

import scipy as sc
import svmutil

# train on training data and test on validation
# y: ndarray, x: csr_matrix
y_train, x_train = svmutil.svm_read_problem(
    '../data/train_svm/svm_input_cc_train_1.in', return_scipy=True)
m = svmutil.svm_train(y_train, x_train)

y_valid, x_valid = svmutil.svm_read_problem(
    '../data/test_svm/svm_input_cc_test_1.in')
p_label, p_acc, p_val = svmutil.svm_predict(y_valid, x_valid, m)
