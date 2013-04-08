import csv
from glob import glob
import logging
import os
from numpy import savetxt
from data import clean_nli_data, nli_train_data_path, nli_train_dataset_fn, nli_dev_data_path, nli_dev_dataset_fn, load_nli_data, nli_test_data_path, nli_test_dataset_fn
from treetagger import simple_tokenize, tree_tag_file

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

   # for fn in glob(os.path.join(nli_train_data_path, '*.txt')) \
   #           + glob(os.path.join(nli_dev_data_path, '*.txt')) \
   #           + glob(os.path.join(nli_test_data_path, '*.txt')):
   #     simple_tokenize(fn, fn + '.tok')
   #     tree_tag_file(fn + '.tok', fn + '.tok.tag')

    clean_nli_data(nli_train_data_path, nli_train_dataset_fn)
    clean_nli_data(nli_dev_data_path, nli_dev_dataset_fn)
    clean_nli_data(nli_test_data_path, nli_test_dataset_fn)
