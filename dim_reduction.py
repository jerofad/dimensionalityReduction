"""
This code is the implementation of dimensionality reduction 
from Johnson Lindestrauss lemma and the paper from Racheal Ward.
"""

import math
import numpy as np
import argparse
import os
import csv


# Args
argparser = argparse.ArgumentParser(
    description="Dimensionality reduction for your own data.")

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to your data",
    default=os.path.join('DATA', 'NIPS.csv'))

argparser.add_argument(
    '-e',
    '--epsilon',
    help='the epsilon value which is the error you can accomodate',
    default=0.1)


def new_dim(n, e):
    # This functions return the number of dimension we would get after
    # the dimensionality reduction
    # n is the number of the rows.
    numerator = 4 * 6 * math.log(n)
    denominator = (3 * math.pow(e, 2)) - (2 * math.pow(e, 3))
    k = numerator/denominator
    print("The new dimension is : " + '{:d}'.format(k))
    return k


def old_dim(data):
    return data.shape[1]


def random_projection(X, e):
    # This function generates the Gaussian Random Matrix
    n = X.shape[0]
    k = new_dim(n, e)
    d = old_dim(X)
    A = np.random.normal(0, 1 / np.sqrt(k), (k, d))
    return A


def new_data(X, e):
    A = random_projection(X,e)
    X_new = np.matmul(A, X)
    print(" The data was reduced from " + X.shape + "to" + X_new.shape)
    return X_new  # here is our new data.

if __name__ == '__main__':
    args = argparser.parse_args()
    data_path = os.path.expanduser(args.data_path)
    epsilon = args.epsilon
    # read the data from file

    X = np.array(list(csv.reader(open(data_path, "r")))).astype("float")
    new_data(X, epsilon)
