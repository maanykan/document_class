# Preprocess data for LIBSVM and scikit-learn Nearest Neighber.  Create input
# data files for LIBSVM in data/train_svm/ and data/test_svm/ directories.

import filter_util as f
# import numpy as np
# import scipy as sp
# from svmutil import *


def normalize_by_three_filters(filename):
    matrix = f.read_data('../data/' + filename + '.data')
    labels = f.read_labels('../data/' + filename + '.labels')
    column_indices_corr_coef = f.sort_ind_corr_coef(matrix, labels)
    column_indices_sig_2_noise, column_indices_t_test = \
        f.sort_ind_signal_2_noise_and_t_test(matrix, labels)
    matrix_norm = f.normalize(matrix)
    matrix_corr_coef_norm = matrix_norm[:, column_indices_corr_coef]
    matrix_sig_2_noise_norm = matrix_norm[:, column_indices_sig_2_noise]
    matrix_t_test_norm = matrix_norm[:, column_indices_t_test]
    return [matrix_corr_coef_norm, matrix_sig_2_noise_norm, matrix_t_test_norm,
            labels]


matrix_corr_coef_norm_train, matrix_sig_2_noise_norm_train, \
    matrix_t_test_norm_train, labels_train = normalize_by_three_filters('dexter_train')
matrix_corr_coef_norm_test, matrix_sig_2_noise_norm_test, \
    matrix_t_test_norm_test, labels_test = normalize_by_three_filters('dexter_valid')

print('\nCreating SVM input files in data/train_svm/ directory:')
f.create_svm_input_files(matrix_corr_coef_norm_train,
                         matrix_sig_2_noise_norm_train,
                         matrix_t_test_norm_train, labels_train, 'train')
print('\nCreating SVM input files in data/test_svm/ directory:')
f.create_svm_input_files(matrix_corr_coef_norm_test,
                         matrix_sig_2_noise_norm_test,
                         matrix_t_test_norm_test, labels_test, 'test')



# matrix_corr_coef = matrix[:, column_indices_corr_coef]
# matrix_sig_2_noise = matrix[:, column_indices_sig_2_noise]
# matrix_t_test = matrix[:, column_indices_t_test]

# np.set_printoptions(precision=2, suppress=True)

# print('indices sorted based on corr coef:\n', column_indices_corr_coef)
# print('indices sorted based on sig2noise:\n', column_indices_sig_2_noise)
# print('indices sorted based on t-test:\n', column_indices_t_test)
# print('matrix:\n', matrix.toarray())
# print('matrix_norm:\n', matrix_norm.toarray())
# print('matrix_corr_coef_norm:\n', matrix_corr_coef_norm.toarray())
# print('matrix_sig_2_noise_norm:\n', matrix_sig_2_noise_norm.toarray())
# print('matrix_t_test_norm:\n', matrix_t_test_norm.toarray())

# print('matrix_corr_coef:\n', matrix_corr_coef.toarray())
# print('matrix_sig_2_noise:\n', matrix_sig_2_noise.toarray())
# print('matrix_t_test:\n', matrix_t_test.toarray())
