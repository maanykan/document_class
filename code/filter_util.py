"""filter_util module contains functions to read feature vectors into a sparse
matrix, read labels into a list, and compute feature selection filters, such as
Pearson correlation coefficient, Signal-to-noise ration and t-test.

Pooya Taherkhani
pt376511 at ohio edu

September 2018

"""

import numpy as np
from scipy.sparse import csc_matrix, vstack, hstack


def read_data(filename):
    """Read Dexter data from feature file and label file into a sparse
    matrix.  For feature selection filename should be
    'dexter_train.data'.
    TAKE IN: a file name that can be prefixed by the relative path
    RETURN: a sparse matrix

    """
    # open feature file, read each line, and fill in the sparse matrix
    i, j, j_max = 0, 0, 0
    data = []
    row_ind = []
    col_ind = []
    with open(filename) as file:
        for line in file:
            for c in line.split():
                j, feature = c.split(':')
                j = int(j)
                data.append(int(feature))
                row_ind.append(i)
                col_ind.append(j - 1)
            if j_max < j:
                j_max = j
            i += 1

    n = i                       # number of examples
    k = j_max                   # number of features

    feature_matrix = csc_matrix((data, (row_ind, col_ind)), shape=(n, k))
    return feature_matrix


def read_labels(filename):
    """Read labels from a label file.
    TAKE IN: labels file name as a string prefixed with relative path
    RETURN: labels list

    """
    labels = []
    with open(filename) as file:
        for line in file:
            label = line.split()[0]
            labels.append(int(label))
    return labels


def sort_ind_corr_coef(feature_matrix, labels):
    """Sort columns of sparse feature matrix based on Pearson correlation
    coefficient.
    TAKE IN: 1. sparse feature matrix  2. labels list
    RETURN: sorted sparse feature matrix based on Pearson correlation
            coefficient.  Columns of matrix represent the most relevant feature
            on the left upto the least relevant on the right.  Sorting is not
            stable as it does not need be.

    """
    n, k = feature_matrix.shape
    y_len = len(labels)
    if n != y_len:
        raise AssertionError('Number of feature vectors and number of labels \
do not match.')
    # y_bar for all features
    y_bar = sum(labels) / y_len
    x_bar_vec = csc_matrix(feature_matrix.sum(0) / k)
    # x_bar for all features
    x_bar_matrix = vstack([x_bar_vec] * n)
    # numerator of correlation coefficient for all features
    x_dif = feature_matrix - x_bar_matrix
    y_dif = np.array(labels) - y_bar
    y_dif.resize(n, 1)
    numerator = x_dif.multiply(y_dif).sum(0)
    # denominator for all features
    denominator = np.asarray(x_dif.multiply(x_dif).sum(0))**0.5 * \
                  (y_dif**2).sum(0)**0.5
    numerator.resize(k)
    denominator.resize(k)
    p_corr_coef = np.asarray(numerator) / denominator
    p_corr_coef[np.isnan(p_corr_coef)] = 0
    p_corr_coef[np.isinf(p_corr_coef)] = 0
    print('p_corr_coef:\n', p_corr_coef)
    p_corr_coef = abs(p_corr_coef)
    # sort columns of feature matrix in descending order based on corr coef
    # value
    p_corr_coef_sorted_indices = np.argsort(-p_corr_coef)
    # return feature_matrix[:, p_corr_coef_sorted_indices]
    return p_corr_coef_sorted_indices


def mean(feature_mat, labels_arr, category):
    """ hint:  number of examples with label 1:  (y == 1).sum()
    indices of examples with label 1:  np.where(y == 1)
    """
    return feature_mat[np.where(labels_arr == category)].sum(0) \
        / (labels_arr == category).sum()


def std_dev(mean_X_sq, mu, kk):
    """mu: must be an ndarray, not a matrix
    kk: number of features in one example
    """
    mean_X_sq.resize(kk)
    mean_X_sq = np.asarray(mean_X_sq)
    return np.sqrt(mean_X_sq - mu**2)


def sort_ind_signal_2_noise_and_t_test(feature_matrix, labels):
    """Sort column indices of feature matrix based on signal to noise filter.

    """
    n, k = feature_matrix.shape
    y = np.asarray(labels)
    # mean of feature X for which Y = 1 for all features
    mu_plus = mean(feature_matrix, y, 1)
    mu_minus = mean(feature_matrix, y, -1)
    # standard deviation
    feature_matrix_squared = feature_matrix.multiply(feature_matrix)
    # mean of X^2 for which Y = 1 for all features
    mean_X_sq_plus = mean(feature_matrix_squared, y, 1)
    mean_X_sq_minus = mean(feature_matrix_squared, y, -1)
    mu_plus.resize(k)
    mu_minus.resize(k)
    mu_plus = np.asarray(mu_plus)
    mu_minus = np.asarray(mu_minus)
    sigma_plus = std_dev(mean_X_sq_plus, mu_plus, k)
    sigma_minus = std_dev(mean_X_sq_minus, mu_minus, k)
    abs_dif_mu = np.abs(mu_plus - mu_minus)
    mu_s2n = abs_dif_mu / (sigma_plus + sigma_minus)  # signal to noise
    mu_s2n[np.isnan(mu_s2n)] = 0
    mu_s2n[np.isinf(mu_s2n)] = 0
    print('mu_s2n:\n', mu_s2n)
    mu_s2n_sorted_indices = np.argsort(-mu_s2n)
    # ============ t-test
    m_plus = (y == 1).sum()
    m_minus = n - m_plus
    t_test = abs_dif_mu / np.sqrt(sigma_plus**2 / m_plus + sigma_minus**2 /
                                  m_minus)
    t_test[np.isnan(t_test)] = 0
    t_test[np.isinf(t_test)] = 0
    print('t_test:\n', t_test)
    t_test_sorted_indices = np.argsort(-t_test)
    s2n_and_t_test_ind = [mu_s2n_sorted_indices, t_test_sorted_indices]
    return s2n_and_t_test_ind


def normalize(matrix):
    """Normalize each feature vector (example) in a feature matrix by dividing
    each vector by its Euclidian norm.
    TAKE IN:  a feature matrix
    RETURN:  a sparse feature matrix with normalized columns

    """
    k = matrix.shape[1]
    x_norm = csc_matrix(np.sqrt(matrix.multiply(matrix).sum(1)))
    x_norm_matrix = hstack([x_norm] * k)
    return csc_matrix(matrix / x_norm_matrix)


# def output_file()
