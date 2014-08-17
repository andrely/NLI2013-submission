import os
from pandas import concat, pandas
from data import load_nli_data, load_nli_frame, NLI_TEST_INDEX_FN, FOLDS_FN, ROOT_PATH, NLI_TEST_PATH, \
    NLI_TEST_DATASET_FN


def get_folds_data(root_path=ROOT_PATH):
    folds = pandas.io.parsers.read_csv(os.path.join(root_path, FOLDS_FN), names=['file', 'fold', 'dataset'])
    train, dev, _ = load_nli_data()
    test_data = load_nli_frame(os.path.join(root_path, NLI_TEST_INDEX_FN),
                               dataset_fn=os.path.join(root_path, NLI_TEST_DATASET_FN))
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

