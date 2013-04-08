import logging

from scipy.sparse import hstack, coo_matrix
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from numpy import sum, zeros, mean, log
from scipy.sparse import spdiags
from sklearn.pipeline import Pipeline

from sk_feature_extractors import FactorIndicators, CollocationCountVectorizer
from tools import flatten


PROMPT_FEATURE_ID = 'prompt'
PROF_FEATURE_ID = 'prof'
CHAR_FEATURE_ID = 'char'
TOKEN_FEATURE_ID = 'token'
SUFFIX_FEATURE_ID = 'suff'
BASIC_FEATURE_ID = 'basic'
TOKEN_COLLOCATION_FEATURE_ID = 'token_coll'
CHAR_COLLOCATION_FEATURE_ID = 'char_coll'
POS_COLLOCATION_FEATURE_ID = 'pos_coll'
SUFFIX_COLLOCATION_FEATURE_ID = 'suff_coll'

sentences_re = re.compile('<s>(.*?)</s>')

def plate_features(dep_features, cond_features):
    plates = []
    cond_card = cond_features.shape[1]

    for i in range(0, cond_card):
        cond = cond_features[:,i]
        cond = spdiags(cond, 0, len(cond), len(cond))

        plates.append(cond * dep_features)

    return hstack(plates)

def parse_nli_text_fields(token, fields=['token', 'lemma', 'tag']):
    token_fields = token.split('||')
    ret = []

    for field in fields:
        if field == 'token':
            ret.append(token_fields[0])
        elif field == 'lemma':
            ret.append(token_fields[1])
        elif field == 'tag':
            ret.append(token_fields[2])

    return ret

def parse_nli_text_segment(text, fields=['token']):
    sentences = re.findall(sentences_re, text)
    sentences = [[parse_nli_text_fields(tok, fields=fields)
                  for tok in sent.split()] for sent in sentences]

    return sentences

def extract_tokens(text):
    sentences = parse_nli_text_segment(text, fields=['token'])
    sentences = [' '.join([tok[0] for tok in sent]) for sent in sentences]

    return "\n".join(sentences)

def extract_suffixes(text, suff_len):
    sentences = parse_nli_text_segment(text, fields=['token'])
    sentences = [' '.join([tok[0][-suff_len:] for tok in sent]) for sent in sentences]

    return "\n".join(sentences)

def extract_pos(text):
    sentences = parse_nli_text_segment(text, fields=['tag'])
    sentences = [' '.join([tok[0] for tok in sent]) for sent in sentences]

    return "\n".join(sentences)

def split_on_lemma(token, lemma):
    if token.find(lemma) == 0:
        suffix = token[len(lemma):]

        return [lemma, suffix]
    else:
        return token

def extract_lemma_suffixes(text):
    sentences = parse_nli_text_segment(text, fields=['token', 'lemma'])
    sentences = [' '.join([' '.join(split_on_lemma(tok[0], tok[1])).strip()
                           for tok in sent]).strip()
                 for sent in sentences]

    return "\n".join(sentences)

def create_category_texts(frame, dataset='train'):
    categories = frame.cat.unique()

    for cat in categories:
        subframe = frame.ix[frame.cat == cat]
        tokens = subframe.tokens.map(extract_tokens).values

        with open("token-%s-%s.txt" % (cat, dataset), 'w') as f:
            for text in tokens:
                f.write(text + "\n")

def avg_sentence_length(text):
    sentences = parse_nli_text_segment(text, fields=['token'])
    sent_lens = [len(sent) for sent in sentences]

    return mean(sent_lens)

def is_word(tok):
    return tok.isalpha()

def avg_word_length(text):
    sentences = parse_nli_text_segment(text, fields=['token'])
    word_lens = flatten([[len(tok[0]) for tok in sent if is_word(tok[0])] for sent in sentences])

    return mean(word_lens)

def basic_measures(tokens):
    x = zeros((len(tokens), 2))

    for i, text in enumerate(tokens):
        x[i, 0] = log(avg_sentence_length(text))
        x[i, 1] = log(avg_word_length(text))

    return coo_matrix(x)

DEFAULT_FEATURES = [TOKEN_FEATURE_ID,
                    CHAR_FEATURE_ID,
                    PROF_FEATURE_ID,
                    PROMPT_FEATURE_ID]

DEFAULT_TOKEN_VECT_ARGS = {'sublinear_tf': True,
                           'smooth_idf': True,
                           'max_df': 0.5,
                           'ngram_range': (1,1),
                           'preprocessor': extract_tokens}

DEFAULT_CHAR_VECT_ARGS = {'sublinear_tf': True,
                          'smooth_idf': True,
                          'max_df': 0.5,
                          'ngram_range': (1, 4),
                          'analyzer': 'char',
                          'preprocessor': extract_tokens}

DEFAULT_SUFF_VECT_ARGS = {'sublinear_tf': True,
                          'smooth_idf': True,
                          'max_df': 0.5,
                          'ngram_range': (1, 1),
                          'preprocessor': lambda text: extract_suffixes(text, suff_len=3)}

DEFAULT_TOKEN_COLL_ARGS = {'preprocessor': extract_tokens,
                           'window': 4}

DEFAULT_POS_COLL_ARGS = {'preprocessor': extract_pos,
                         'window': 1}

DEFAULT_CHAR_COLL_ARGS = {'preprocessor': extract_tokens,
                          'window': 3,
                          'ngram_range': (2,4),
                          'analyzer': 'char'}

DEFAULT_SUFF_COLL_ARGS = {'preprocessor': lambda text: extract_suffixes(text, suff_len=3),
                          'window': 4}

def merge_default_args(default_args, args):
    if not args:
        return default_args

    for k, v in args.items():
        default_args[k] = v

    return default_args

class FeaturePipeline(BaseEstimator, TransformerMixin):
    def __init__(self, features=DEFAULT_FEATURES, token_vect_args=None, suff_vect_args=None,
                 char_vect_args=None, token_coll_args=None, pos_coll_args=None, char_coll_args=None,
                 suff_coll_args=None):
        self.features = features

        self.token_ex = TfidfVectorizer(**merge_default_args(DEFAULT_TOKEN_VECT_ARGS, token_vect_args))
        self.char_ex = TfidfVectorizer(**merge_default_args(DEFAULT_CHAR_VECT_ARGS, char_vect_args))
        self.suff_ex = TfidfVectorizer(**merge_default_args(DEFAULT_SUFF_VECT_ARGS, suff_vect_args))
        self.prof_ex = FactorIndicators()
        self.prompt_ex = FactorIndicators()

        self.token_coll_ex = Pipeline([('collocation',
                                        CollocationCountVectorizer(**merge_default_args(DEFAULT_TOKEN_COLL_ARGS, token_coll_args))),
                                       ('tfidf',
                                        TfidfTransformer(sublinear_tf=True))])
        self.pos_coll_ex = Pipeline([('collocation',
                                      CollocationCountVectorizer(**merge_default_args(DEFAULT_POS_COLL_ARGS, pos_coll_args))),
                                     ('tfidf',
                                      TfidfTransformer(sublinear_tf=True))])

        # TODO test tfidf
        self.char_coll_ex = Pipeline([('collocation',
                                       CollocationCountVectorizer(**merge_default_args(DEFAULT_CHAR_COLL_ARGS, char_coll_args))),
                                      ('tfiidf',
                                       TfidfTransformer(sublinear_tf=True))])


        self.suff_coll_ex = Pipeline([('collocation',
                                       CollocationCountVectorizer(**merge_default_args(DEFAULT_SUFF_COLL_ARGS, suff_coll_args))),
                                      ('tfidf',
                                       TfidfTransformer(sublinear_tf=True))])

    def _check_fitted(self):
        # TODO update this
        return (not hasattr(self.token_ex, 'vocabulary_') or len(self.token_ex.vocabulary_) == 0) and \
               (not hasattr(self.char_ex , 'vocabulary_') or len(self.char_ex.vocabulary_) == 0) and \
               self.prof_ex._check_fitted()

    def fit(self, frame):
        if TOKEN_FEATURE_ID in self.features:
            logging.info("Generating token features")
            self.token_ex.fit(frame['tokens'].values)
            logging.info("Generated %s token features" % len(self.token_ex.get_feature_names()))

        if CHAR_FEATURE_ID in self.features:
            logging.info("Generating char ngram features")
            self.char_ex.fit(frame['tokens'].values)
            logging.info("Generated %s char ngram features" % len(self.char_ex.get_feature_names()))

        if SUFFIX_FEATURE_ID in self.features:
            logging.info("Generating suffix features")
            self.suff_ex.fit(frame['tokens'].values)
            logging.info("Generated %s suffix features" % len(self.suff_ex.get_feature_names()))

        if PROF_FEATURE_ID in self.features:
            logging.info("Generating proflevel features")
            self.prof_ex.fit(frame['proflevel'].values)
            logging.info("Generated %d proflevel features" % sum(self.prof_ex.non_value_mask_))

        if PROMPT_FEATURE_ID in self.features:
            logging.info("Generating prompt features")
            self.prompt_ex.fit(frame['prompt'].values)
            logging.info("Generated %d prompt features" % sum(self.prompt_ex.non_value_mask_))

        if TOKEN_COLLOCATION_FEATURE_ID in self.features:
            logging.info("Generating token collocation features")
            self.token_coll_ex.fit(frame['tokens'].values)
            logging.info("Generated %d token collocation features" % len(self.token_coll_ex.steps[0][1].vocabulary_))

        if POS_COLLOCATION_FEATURE_ID in self.features:
            logging.info("Generating POS collocation features")
            self.pos_coll_ex.fit(frame['tokens'].values)
            logging.info("Generated %d POS collocation features" % len(self.pos_coll_ex.steps[0][1].vocabulary_))

        if CHAR_COLLOCATION_FEATURE_ID in self.features:
            logging.info("Generating char n-gram collocation features")
            self.char_coll_ex.fit(frame['tokens'].values)
            logging.info("Generated %d char n-gram collocation features" % len(self.char_coll_ex.steps[0][1].vocabulary_))

        if SUFFIX_COLLOCATION_FEATURE_ID in self.features:
            logging.info("Generating suffix collocation features")
            self.suff_coll_ex.fit(frame['tokens'].values)
            logging.info("Generated %d suffix collocation features" % len(self.suff_coll_ex.steps[0][1].vocabulary_))

        return self

    def transform(self, frame):
        feature_sets = []

        if BASIC_FEATURE_ID in self.features:
            feature_sets.append(basic_measures(frame.tokens.values))
            logging.info("Generated %d basic features" % feature_sets[0].shape[1])
        if TOKEN_FEATURE_ID in self.features:
            feature_sets.append(self.token_ex.transform(frame['tokens'].values))
        if SUFFIX_FEATURE_ID in self.features:
            feature_sets.append(self.suff_ex.transform(frame['tokens'].values))
        if CHAR_FEATURE_ID in self.features:
            feature_sets.append(self.char_ex.transform(frame['tokens'].values))
        if PROF_FEATURE_ID in self.features:
            feature_sets.append(self.prof_ex.transform(frame['proflevel'].values))
        if PROMPT_FEATURE_ID in self.features:
            feature_sets.append(self.prompt_ex.transform(frame['prompt'].values))
        if TOKEN_COLLOCATION_FEATURE_ID in self.features:
            feature_sets.append(self.token_coll_ex.transform(frame['tokens'].values))
        if POS_COLLOCATION_FEATURE_ID in self.features:
            feature_sets.append(self.pos_coll_ex.transform(frame['tokens'].values))
        if CHAR_COLLOCATION_FEATURE_ID in self.features:
            feature_sets.append(self.char_coll_ex.transform(frame['tokens'].values))
        if SUFFIX_COLLOCATION_FEATURE_ID in self.features:
            feature_sets.append(self.suff_coll_ex.transform(frame['tokens'].values))

        x = hstack(feature_sets)

        logging.info("Generated %s features" % x.shape[1])

        return x
