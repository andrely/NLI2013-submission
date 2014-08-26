from collections import Counter
from itertools import combinations
import numbers
from numpy import unique, isnan, ones, zeros
import numpy
import scipy.sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import VectorizerMixin, CountVectorizer
from tools import split_sentence_boundaries, sorted_tuple

class CollocationCountVectorizer(CountVectorizer):
    def __init__(self, window=None, sentence_splitter="\n", directional=False, **args):
        self.window = window
        self.sentence_splitter = sentence_splitter
        self.directional = directional
        CountVectorizer.__init__(self, **args)

    def _ngram_collocations(self, sents):
        collocations = []

        for sent in sents:
            sent = self._white_spaces.sub(u" ", sent)

            if self.analyzer == 'word':
                sent = sent.split()

            sent_len = len(sent)
            min_n, max_n = self.ngram_range

            for n in xrange(min_n, min(max_n + 1, sent_len + 1)):
                ngrams = []

                for i in xrange(sent_len - n + 1):
                    ngrams.append(sent[i: i + n])

                for i, j in combinations(xrange(len(ngrams)), 2):
                    if self.window and abs(i - j) > self.window:
                        continue

                    collocation = (ngrams[i], ngrams[j])

                    if not self.directional:
                        collocation = sorted_tuple(collocation)

                    collocations.append(collocation)

            return collocations

    def _collocations(self, sents):
        collocations = []

        for sent in sents:
            sent = self._white_spaces.sub(u" ", sent)

            if self.analyzer == 'word':
                sent = sent.split()

            sent_len = len(sent)

            for i, j in combinations(xrange(sent_len), 2):
                if self.window and abs(i - j) > self.window:
                    continue

                collocation = (sent[i], sent[j])

                if not self.directional:
                    collocation = sorted_tuple(collocation)

                collocations.append(collocation)

        return collocations

    def _split_sentences(self, raw_document):
        if self.sentence_splitter == None:
            return raw_document
        elif callable(self.sentence_splitter):
            return self.sentence_splitter(raw_document)
        else:
            return raw_document.split(self.sentence_splitter)

    def _analyze(self, raw_document):
        if self.preprocessor:
            raw_document = self.preprocessor(raw_document)

        # split sentences
        sents = self._split_sentences(raw_document)

        if self.ngram_range == (1, 1):
            return self._collocations(sents)
        else:
            return self._ngram_collocations(sents)

    def fit_transform(self, raw_documents, y=None):
        collocations_per_doc = []
        collocation_counts = Counter()

        for doc in raw_documents:
            doc_collocation_count = Counter(self._analyze(doc))
            collocation_counts.update(doc_collocation_count)

            collocations_per_doc.append(doc_collocation_count)


        # Adapted from base class CountVectorizer
        n_doc = len(collocations_per_doc)
        max_df = self.max_df
        min_df = self.min_df

        max_doc_count = (max_df
                         if isinstance(max_df, numbers.Integral)
                         else max_df * n_doc)
        min_doc_count = (min_df
                         if isinstance(min_df, numbers.Integral)
                         else min_df * n_doc)

        # filter out stop words: terms that occur in almost all documents
        if max_doc_count < n_doc or min_doc_count > 1:
            stop_words = set(t for t, dc in collocation_counts.iteritems()
                if dc > max_doc_count or dc < min_doc_count)
        else:
            stop_words = set()

        # list the terms that should be part of the vocabulary
        if self.max_features is None:
            terms = set(collocation_counts) - stop_words
        else:
            # extract the most frequent terms for the vocabulary
            terms = set()
            for t, tc in collocation_counts.most_common():
                if t not in stop_words:
                    terms.add(t)
                if len(terms) >= self.max_features:
                    break

        # store map from term name to feature integer index: we sort the term
        # to have reproducible outcome for the vocabulary structure: otherwise
        # the mapping from feature name to indices might depend on the memory
        # layout of the machine. Furthermore sorted terms might make it
        # possible to perform binary search in the feature names array.
        vocab = dict(((t, i) for i, t in enumerate(sorted(terms))))
        if not vocab:
            raise ValueError("empty vocabulary; training set may have"
                             " contained only stop words or min_df (resp. "
                             "max_df) may be too high (resp. too low).")
        self.vocabulary_ = vocab

        return self._term_count_dicts_to_matrix(collocations_per_doc)

    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents)

        return self

    def transform(self, raw_documents):
        if not hasattr(self, 'vocabulary_') or len(self.vocabulary_) == 0:
            raise ValueError("Vocabulary wasn't fitted or is empty!")

        collocation_counts_per_doc = (Counter(self._analyze(doc)) for doc in raw_documents)
        return self._term_count_dicts_to_matrix(collocation_counts_per_doc)

    def _term_count_dicts_to_matrix(self, term_count_dicts):
        i_indices = []
        j_indices = []
        values = []
        vocabulary = self.vocabulary_

        for i, term_count_dict in enumerate(term_count_dicts):
            for term, count in term_count_dict.iteritems():
                j = vocabulary.get(term)
                if j is not None:
                    i_indices.append(i)
                    j_indices.append(j)
                    values.append(count)
            # free memory as we go
            term_count_dict.clear()

        shape = (i + 1, max(vocabulary.itervalues()) + 1)
        spmatrix = scipy.sparse.coo_matrix((values, (i_indices, j_indices)),
                                 shape=shape, dtype=self.dtype)
        if self.binary:
            spmatrix.data.fill(1)
        return spmatrix

class FactorIndicators(BaseEstimator, TransformerMixin):
    def __init__(self, neg_label=0, pos_label=1, remove_nan=True, non_values=[]):
        if neg_label >= pos_label:
            raise ValueError("neg_label must be strictly less than pos_label.")

        self.neg_label = neg_label
        self.pos_label = pos_label
        self.remove_nan = remove_nan
        self.non_values = non_values

    def _check_fitted(self):
        if not hasattr(self, "classes_") or not hasattr(self, "non_value_mask_"):
            raise ValueError("LabelBinarizer was not fitted yet.")

    def fit(self, y):
        self.classes_ = unique(y)
        self.non_value_mask_ = ones(len(self.classes_))

        for i, c in enumerate(self.classes_):
            if (self.remove_nan and
                isinstance(c, (float, long, int, complex)) and
                isnan(c)) or \
               c in self.non_values:

                self.non_value_mask_[i] = 0

        return self

    def transform(self, y):
        self._check_fitted()

        Y = zeros((len(y), len(self.classes_)), dtype=numpy.int)
        Y += self.neg_label

        for i, k in enumerate(self.classes_):
            Y[y == k, i] = self.pos_label

        Y = Y.compress(self.non_value_mask_, axis=1)

        return Y
