from pandas import concat, pandas
from data import load_nli_data, load_nli_frame, nli_test_index_fn, folds_fn, nli_test_path


def get_folds_data():
    folds = pandas.io.parsers.read_csv(folds_fn, names=['file', 'fold', 'dataset'])
    train, dev = load_nli_data()
    test_data = load_nli_frame(nli_test_index_fn, dataset_path=nli_test_path)
    data = concat((train, dev, test_data))

    folds_data = []
    file_list = []

    for i in range(10):
        files = folds[folds.fold == (i+1)]
        train_files = files[files.dataset == 'train'].file
        test_files = files[files.dataset == 'test'].file

        file_list.append(sorted(train_files.values))

        train = data[data.file.isin(train_files)]
        test = data[data.file.isin(test_files)]

        folds_data.append((train, test))

    return folds_data

