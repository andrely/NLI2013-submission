from glob import glob
import logging
import os
import sys

from data import clean_nli_data, NLI_TRAIN_DATA_PATH, NLI_TRAIN_DATASET_FN, NLI_DEV_DATA_PATH, NLI_DEV_DATASET_FN, \
    NLI_TEST_DATA_PATH, NLI_TEST_DATASET_FN
from treetagger import simple_tokenize, tree_tag_file


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv == 2):
        root_path = sys.argv[1]
    else:
        logging.error("Path to dataset is a required argument.")
        sys.exit(1)

    for fn in glob(os.path.join(os.path.join(root_path, NLI_TRAIN_DATA_PATH), '*.txt')) \
            + glob(os.path.join(os.path.join(root_path, NLI_DEV_DATA_PATH), '*.txt')) \
            + glob(os.path.join(os.path.join(root_path, NLI_TEST_DATA_PATH), '*.txt')):
        simple_tokenize(fn, fn + '.tok')
        tree_tag_file(fn + '.tok', fn + '.tok.tag')

    clean_nli_data(os.path.join(root_path, NLI_TRAIN_DATA_PATH),
                   os.path.join(root_path, NLI_TRAIN_DATASET_FN))
    clean_nli_data(os.path.join(root_path, NLI_DEV_DATA_PATH),
                   os.path.join(root_path, NLI_DEV_DATASET_FN))
    clean_nli_data(os.path.join(root_path, NLI_TEST_DATA_PATH),
                   os.path.join(root_path, NLI_TEST_DATASET_FN))
