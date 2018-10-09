import filter_util as f
# import numpy as np
# import scipy as sp
# from smut import *

matrix = f.read_data('../data/dexter_train_copy.data')
labels = f.read_labels('../data/dexter_train_copy.labels')
column_indices_corr_coef = f.sort_ind_corr_coef(matrix, labels)
column_indices_sig_2_noise, \
   column_indices_t_test = f.sort_ind_signal_2_noise_and_t_test(matrix, labels)
matrix_norm = f.normalize(matrix)
matrix_corr_coef_norm = matrix_norm[:, column_indices_corr_coef]
matrix_sig_2_noise_norm = matrix_norm[:, column_indices_sig_2_noise]
matrix_t_test_norm = matrix_norm[:, column_indices_t_test]

f.create_svm_input_files(matrix_corr_coef_norm, matrix_sig_2_noise_norm, \
                         matrix_t_test_norm, labels)

# N = [1, 5, 10, 20, 50] + [n for n in range(100, 1000, 100)] + \
#     [n for n in range(1000, 21000, 1000)]

# N = [1, 4, 7, 10]

# print('matrix:\n', matrix_corr_coef_norm.toarray())

# for n in N:
#     matrix_cc_chopped = matrix_corr_coef_norm[:, range(0, n)]
#     matrix_s2n_chopped = matrix_sig_2_noise_norm[:, range(0, n)]
#     matrix_tt_chopped = matrix_t_test_norm[:, range(0, n)]
#     f.svm_input_file(matrix_cc_chopped, labels, 'cc')
#     f.svm_input_file(matrix_sig_2_noise_norm, labels, 's2n')
#     f.svm_input_file(matrix_t_test_norm, labels, 'tt')



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
