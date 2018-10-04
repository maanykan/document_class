"""Preprocess data for LIBSVM package

Merge two files of Dexter dataset, one containing attributes and another
containing labels, into a single file that fits the format required by LIBSVM

Question 8 -- Feature Selection
Project 3
CS 6830 -- Machine Learning

Pooya Taherkhani
July 2018

"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('file_type')
args = parser.parse_args()

if args.file_type != 'train' and args.file_type != 'valid':
    quit("file_type argument must be either 'train' or 'valid'")

rel_path = '../data/'
filename = rel_path + 'dexter_' + args.file_type
data_filename = filename + '.data'
labels_filename = filename + '.labels'
svm_filename = rel_path + 'svm_data.' + args.file_type

with open(data_filename) as data_file:
    with open(labels_filename) as labels_file:
        with open(svm_filename, 'w') as svm_data_file:
            for data_line, label_line in zip(data_file, labels_file):
                label = label_line.split('\n')[0]
                svm_data_file.write(label)
                svm_data_file.write(' ')
                svm_data_file.write(data_line)

print(svm_filename, 'was created.')
