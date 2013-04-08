from BeautifulSoup import BeautifulSoup
import os
import numpy as np

def ensure_2d_matrix(m, axis = 0):
    if np.ndim(m) == 1:
        if axis == 0:
            m.shape = len(m), 1
        elif axis == 1:
            m.shape = 1, len(m)
        else:
            raise ValueError
    elif np.ndim(m) > 2:
        raise ValueError

    return m

def add_bias(x):
    N = x.shape[0]

    return np.hstack((-np.ones((N, 1)), x))

def labels_to_indicators(y, labels = None):
    if labels == None:
        labels = np.unique(y).tolist()

    indicators = np.zeros((len(y), len(labels)))

    for i, label in enumerate(y):
        indicators[i, labels.index(label)] = 1

    return indicators

def indicators_to_labels(y, labels):
    y_max = np.argmax(y, axis=1)
    y_label = [None] * y.shape[0]

    for i, idx in enumerate(y_max):
        y_label[i] = labels[idx]

    return y_label

def flatten(l):
    return [item for sublist in l for item in sublist]

def get_root_path():
    py_path = get_root_python_path()
    root_path = os.path.join(py_path, '..')

    return os.path.abspath(root_path)

def get_root_python_path():
    return os.path.dirname(os.path.realpath(__file__))

def makedirs(path, mode=0777):
    """
    a version of os.makedirs which raises no exception when the dir already
    exists
    """
    try:
        os.makedirs(path, mode)
    except OSError, inst:
        if inst.errno == 17:
            # dir already exists
            pass

def split_sentence_boundaries(sentences_string):
    sents = []

    doc = BeautifulSoup(sentences_string)

    for s in doc.findAll('s'):
        sents.append(s.getText())

    return sents

def sorted_tuple(a_tuple, **args):
    return tuple(sorted(a_tuple, **args))
