import csv
from glob import glob
import logging
import os

import pandas


INDEX_NAMES = ['file', 'prompt', 'cat', 'proflevel']

# root_path = get_root_path()
ROOT_PATH = "/Users/stinky/Work/ML/data"

NLI_TRAIN_PATH = 'NLI_2013_Training_Data/'
NLI_DEV_PATH = 'NLI_2013_Development_Data/'
NLI_TEST_PATH = 'NLI_2013_Test_Data/'

NLI_DATA_DIR = 'tokenized'

NLI_TRAIN_DATA_PATH = os.path.join(NLI_TRAIN_PATH, NLI_DATA_DIR)
NLI_DEV_DATA_PATH = os.path.join(NLI_DEV_PATH, NLI_DATA_DIR)
NLI_TEST_DATA_PATH = os.path.join(NLI_TEST_PATH, NLI_DATA_DIR)

NLI_TRAIN_INDEX_FN = os.path.join(NLI_TRAIN_PATH, 'index-training.csv')
NLI_DEV_INDEX_FN = os.path.join(NLI_DEV_PATH, 'index-dev.csv')
NLI_TEST_INDEX_FN = os.path.join(NLI_TEST_PATH, 'index_test_public_with_L1s.csv')

NLI_TRAIN_DATASET_FN = 'nli_train_dataset.txt'
NLI_DEV_DATASET_FN = 'nli_dev_dataset.txt'
NLI_TEST_DATASET_FN = 'nli_test_dataset.txt'

FOLDS_FN = os.path.join(NLI_TEST_PATH, 'folds_ids_public.csv')

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

def load_nli_frame(index_fn, dataset_fn=None, dataset_path=None, gold=True):
    if not dataset_fn and not dataset_path:
        raise ValueError

    if dataset_fn and dataset_path:
        raise ValueError

    data = None

    if dataset_fn:
        data = load_text_data_csv(dataset_fn)

    if dataset_path:
        data = load_text_data(dataset_path)

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

def load_text_data_csv(dataset_fn):
    logging.info("Reading dataset from %s" % dataset_fn)
    if not os.path.exists(dataset_fn):
        logging.warn("Dataset does not exist %s" % dataset_fn)
        raise RuntimeError

    data = pandas.io.parsers.read_csv(dataset_fn, sep=",", quotechar='|',
                                      names=['file', 'tokens'])
    logging.info("%d rows read" % len(data))

    return data

def load_text_data(dataset_path):
    text_fn_glob = os.path.join(dataset_path, os.path.join(NLI_DATA_DIR, '*.txt'))
    logging.info("Reading dataset from %s" % text_fn_glob)
    text_fns = glob(text_fn_glob)

    data = []

    for text_fn in text_fns:
        filename = os.path.basename(text_fn)
        text = ""

        with open(text_fn, 'r') as f:
            for line in f.readlines():
                text += "<s>%s</s>n" % line.strip()

        data.append((filename, text))

    data = pandas.DataFrame(data, columns=['file', 'tokens'])

    logging.info("%d rows read" % len(data))

    return data

def get_dev_split_indices(frame):
    train = frame.proflevel.index[frame.proflevel.apply(lambda x: isinstance(x, str))]
    dev = frame.proflevel.index[frame.proflevel.apply(lambda x: isinstance(x, float))]

    return train, dev

def load_nli_data(root_path=ROOT_PATH):
    train = load_nli_frame(os.path.join(root_path, NLI_TRAIN_INDEX_FN),
                           dataset_path=os.path.join(root_path, NLI_TRAIN_PATH))
    dev = load_nli_frame(os.path.join(root_path, NLI_DEV_INDEX_FN),
                         dataset_path=os.path.join(root_path, NLI_DEV_PATH))
    test = load_nli_frame(os.path.join(root_path, NLI_TEST_INDEX_FN),
                          dataset_path=os.path.join(root_path, NLI_TEST_PATH))

    return train, dev, test
