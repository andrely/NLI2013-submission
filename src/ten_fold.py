import logging
from pandas import concat, pandas
from sklearn.cross_validation import KFold
from data import load_nli_data, load_nli_frame, nli_test_dataset_fn, nli_test_index_fn, folds_fn

def get_folds_data():
    folds = pandas.io.parsers.read_csv(folds_fn, names=['file', 'fold', 'dataset'])
    train, dev = load_nli_data()
    test = load_nli_frame(nli_test_dataset_fn, nli_test_index_fn)
    data = concat((train, dev, test))

    folds_data = []

    for i in range(10):
        files = folds[folds.fold == (i+1)].file
        train = data[data.file.isin(files)]
        test = data[-data.file.isin(files)]

        folds_data.append((train, test))

    return folds_data
