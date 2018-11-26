from __future__ import print_function

import gzip
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
# import theano
from keras.utils.data_utils import get_file


def load_data(data_file, url):
    """loads the data from the gzip pickled files, and converts to numpy arrays"""
    print('loading data ...')
    path = get_file(data_file, origin=url)
    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = load_pickle(f)
    f.close()

    train_set_x, train_set_y = make_numpy_array(train_set)
    valid_set_x, valid_set_y = make_numpy_array(valid_set)
    test_set_x, test_set_y = make_numpy_array(test_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


def make_numpy_array(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    # data_x = np.asarray(data_x, dtype=theano.config.floatX)
    data_x = np.asarray(data_x, dtype='float32')
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y


def svm_classify(data, C, view=1):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    if view == 1:
        train_data, _, train_label = data[0]
        valid_data, _, valid_label = data[1]
        test_data, _, test_label = data[2]
    elif view==2:
        _, train_data, train_label = data[0]
        _, valid_data, valid_label = data[1]
        _, test_data, test_label = data[2]
    else:
        raise ValueError('Invalid View')


    print('training SVM...')
    #clf = LogisticRegression(C=C, multi_class='multinomial')
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_data, train_label)

    test_p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, test_p)
    valid_p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, valid_p)

    return [test_acc, valid_acc], [test_p, valid_p], [test_label, valid_label]


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret

