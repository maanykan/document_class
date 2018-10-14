# Preprocess data for LIBSVM and scikit-learn Nearest Neighber

import filter_util as f
# import numpy as np
# import scipy as sp
# from svmutil import *

matrix = f.read_data('../data/dexter_train_copy.data')
labels = f.read_labels('../data/dexter_train_copy.labels')
column_indices_corr_coef = f.sort_ind_corr_coef(matrix, labels)
column_indices_sig_2_noise, \
   column_indices_t_test = f.sort_ind_signal_2_noise_and_t_test(matrix, labels)
matrix_norm = f.normalize(matrix)
matrix_corr_coef_norm = matrix_norm[:, column_indices_corr_coef]
matrix_sig_2_noise_norm = matrix_norm[:, column_indices_sig_2_noise]
matrix_t_test_norm = matrix_norm[:, column_indices_t_test]

f.create_svm_input_files(matrix_corr_coef_norm, matrix_sig_2_noise_norm,
                         matrix_t_test_norm, labels, 'train')
f.create_svm_input_files(matrix_corr_coef_norm, matrix_sig_2_noise_norm,
                         matrix_t_test_norm, labels, 'test')


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
