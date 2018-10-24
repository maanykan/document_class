# Document Classifier #

*Ongoing project.*

Determine if a document is realted to a particular subject (in this case
'corporate acquisition') using support vector machines and nearest neighbor
algorithm.

# Installation #

## Download LIBSVM ##

First, download the LIBSVM package as we have used the Python module of that
package for training and testing support vector machines.

## Compile LIBSVM ##

After unziping LIBSVM, change directory to 'libsvm-3.22/python/' (yours may
have a different version number), and type 'make'.

## Copy LIBSVM executable and Python code ##

Now copy 'svm.py', 'svmutil.py', and 'commonutil.py' from the 'python'
directory of LIBSVM into our 'code' directory.  Also, copy the executable
'libsvm.so.2' from 'libsvm-3.22' to our main directory (one above 'code'
directory).

## Run project ##

Now to run our Python scripts change directory to 'code'.  Then,

```bash
$ python3 create_svm_files.py
$ python3 test_svm.py
```