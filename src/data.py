import csv
from glob import glob
import logging
import os

import pandas

from tools import get_root_path

INDEX_NAMES = ['file', 'prompt', 'cat', 'proflevel']

# root_path = get_root_path()
root_path = "/Users/stinky/Work/ML/"

nli_train_path = os.path.join(root_path, 'data/NLI_2013_Training_Data/')
nli_dev_path = os.path.join(root_path, 'data/NLI_2013_Development_Data/')
nli_test_path = os.path.join(root_path, 'data/NLI_2013_Test_Data/')

NLI_DATA_DIR = 'tokenized'

nli_train_data_path = os.path.join(nli_train_path, NLI_DATA_DIR)
nli_dev_data_path = os.path.join(nli_dev_path, NLI_DATA_DIR)
nli_test_data_path = os.path.join(nli_test_path, NLI_DATA_DIR)

nli_train_index_fn = os.path.join(nli_train_path, 'index-training.csv')
nli_dev_index_fn = os.path.join(nli_dev_path, 'index-dev.csv')
nli_test_index_fn = os.path.join(nli_test_path, 'index-test.csv')

nli_train_dataset_fn = 'nli_train_dataset.txt'
nli_dev_dataset_fn = 'nli_dev_dataset.txt'
nli_test_dataset_fn = 'nli_test_dataset.txt'

def clean_nli_data(path, dataset_fn):
    logging.info("Preprocessing data set in %s" % path)
    logging.info("Writing dataset to %s" % dataset_fn)

    with open(dataset_fn, 'w') as f:
        f_csv = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
        text_fns = sorted(glob(os.path.join(path, '*.txt')))
        tagged_fns = sorted(glob(os.path.join(path, '*.txt.tok.tag')))

        for text_fn, tag_fn in zip(text_fns, tagged_fns):
            logging.debug("Reading %s and %s" % (text_fn, tag_fn))
            sentences = []

            with open(text_fn, 'r') as f:
                for line in f.readlines():
                    sentences.append(line.strip())

            tt_sentences = []

            with open(tag_fn, 'r') as f:
                tt_sent = []
                for line in f.readlines():
                    line = line.strip()

                    if line in ['<s>', '</s>']:
                        if tt_sent:
                            tt_sentences.append(tt_sent)

                        tt_sent = []
                    else:
                        toks = line.split('\t')
                        lemma = toks.pop().strip()
                        tag = toks.pop().strip()

                        tt_sent.append("_".join(toks).strip() + "||" + lemma + "||" + tag)

            sentences = [sent for sent in sentences if len(sent.strip()) > 0]
            tt_sentences = [tt_sent for tt_sent in tt_sentences if len(tt_sent) > 0]

            for sent, tt_sent in zip(sentences, tt_sentences):
                if not len(sent.split()) == len(tt_sent):
                    logging.warn("Inconsistent sentence lengths in %s", text_fn)

            tt_sentences = ['<s>' + " ".join(sent) + '</s>' for sent in tt_sentences]
            row = [os.path.basename(text_fn), ''.join(tt_sentences)]

            f_csv.writerow(row)

def load_nli_frame(dataset_fn, index_fn, gold=True):
    logging.info("Reading dataset from %s" % dataset_fn)
    if not os.path.exists(dataset_fn):
        logging.warn("Dataset does not exist %s" % dataset_fn)
        raise RuntimeError

    data = pandas.io.parsers.read_csv(dataset_fn, sep=",", quotechar='|',
        names=['file', 'tokens'])
    logging.info("%d rows read" % len(data))

    logging.info("Reading metadata from %s" % index_fn)
    index_names = INDEX_NAMES

    if not gold:
        index_names.remove('cat')

    metadata = pandas.io.parsers.read_csv(index_fn, names=index_names)
    logging.info("%d rows read" % len(metadata))

    logging.info("Merging data")
    data = pandas.merge(data, metadata, on=['file'], how='outer')
    logging.info("%d rows in final frame" % len(data))

    return data

def get_dev_split_indices(frame):
    train = frame.proflevel.index[frame.proflevel.apply(lambda x: isinstance(x, str))]
    dev = frame.proflevel.index[frame.proflevel.apply(lambda x: isinstance(x, float))]

    return train, dev

def load_nli_data():
    train = load_nli_frame(nli_train_dataset_fn, nli_train_index_fn)
    dev = load_nli_frame(nli_dev_dataset_fn, nli_dev_index_fn)

    return train, dev