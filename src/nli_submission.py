import csv
import logging
from optparse import OptionParser
from numpy import mean, std
from pandas import concat
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from data import load_nli_data, load_nli_frame, nli_test_dataset_fn, nli_test_index_fn
from features import DEFAULT_FEATURES, TOKEN_COLLOCATION_FEATURE_ID, FeaturePipeline, SUFFIX_COLLOCATION_FEATURE_ID, extract_suffixes, TOKEN_FEATURE_ID
from ten_fold import get_folds_data

feature_set_map = {
    'system1': {'features': DEFAULT_FEATURES + [TOKEN_COLLOCATION_FEATURE_ID, SUFFIX_COLLOCATION_FEATURE_ID],
                'token_coll_args': {'window': 1,
                                    'directional': True},
                'char_vect_args': {'ngram_range': (3, 6)},
                'suff_coll_args': {'preprocessor': lambda text: extract_suffixes(text, suff_len=4),
                                   'directional': True,
                                   'window': 1}},
    'system2': {'features': DEFAULT_FEATURES + [TOKEN_COLLOCATION_FEATURE_ID, SUFFIX_COLLOCATION_FEATURE_ID],
                'token_vect_args': {'min_df': 5},
                'char_vect_args': {'min_df': 5, 'ngram_range': (3, 6)},
                'token_coll_args': {'min_df': 5,
                                    'window': 1,
                                    'directional': True},
                'suff_coll_args': {'min_df': 5,
                                   'preprocessor': lambda text: extract_suffixes(text, suff_len=4),
                                   'directional': True,
                                   'window': 1}},
    'system3': {'features': DEFAULT_FEATURES + [TOKEN_COLLOCATION_FEATURE_ID, SUFFIX_COLLOCATION_FEATURE_ID],
                'token_vect_args': {'min_df': 10},
                'char_vect_args': {'min_df': 10,
                                   'ngram_range': (3, 6)},
                'token_coll_args': {'min_df': 10,
                                    'window': 1,
                                    'directional': True},
                'suff_coll_args': {'min_df': 10,
                                   'preprocessor': lambda text: extract_suffixes(text, suff_len=4),
                                   'directional': True,
                                   'window': 1}},
    'system4': {'features': DEFAULT_FEATURES + [TOKEN_COLLOCATION_FEATURE_ID, SUFFIX_COLLOCATION_FEATURE_ID],
                'token_vect_args': {'min_df': 10},
                'char_vect_args': {'min_df': 10,
                                   'ngram_range': (1, 7)},
                'token_coll_args': {'min_df': 5,
                                    'window': 1,
                                    'directional': True},
                'suff_coll_args': {'min_df': 5,
                                   'preprocessor': lambda text: extract_suffixes(text, suff_len=4),
                                   'directional': True,
                                   'window': 1}},
    'basic': {'features': [TOKEN_FEATURE_ID]}
}

def predict(model, input, input_frame, out_fn):
    out = model.predict(input)
    out_label = lb.inverse_transform(out.astype(int))
    logging.info("Writing predictions to %s" % out_fn)
    with open(out_fn, 'w') as f:
        csv_out = csv.writer(f)

        for i, label in enumerate(out_label):
            csv_out.writerow([input_frame.file.ix[i], label])


def do_fixed_folds():
    logging.info("doing 10-fold")
    scores = []
    for i, (tf_train, tf_test) in enumerate(get_folds_data()):
        fold = i + 1
        logging.info("Fold %d" % fold)

        ex = FeaturePipeline(**feature_args)
        tf_x_train = ex.fit_transform(tf_train)
        tf_y_train = lb.transform(tf_train.cat.values)

        tf_x_test = ex.transform(tf_test)
        tf_y_test = lb.transform(tf_test.cat.values)

        model = GridSearchCV(LinearSVC(), {'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]},
                             verbose=1, n_jobs=jobs)
        model.fit(tf_x_train, tf_y_train)

        out = model.predict(tf_x_test)

        score = accuracy_score(tf_y_test, out)
        logging.info("Score %s" % score)
        scores.append(score)

    print "System 1 10-fold mean: %f, stddev: %f" % (mean(scores), std(scores))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = OptionParser()

    parser.add_option("-n", "--n-jobs", help="Number of concurrent jobs run during CV and grid search.",
                      default=1)
    parser.add_option("-f", "--feature-sets", help="Comma separated list of feature ids",
                      default="basic")

    opts, args = parser.parse_args()

    jobs = int(opts.n_jobs)
    logging.info("Running %d concurrent jobs")

    feature_sets = opts.feature_sets.lower().split(",")
    logging.info("Running feature sets %s" % ", ".join(feature_sets))

    train, dev = load_nli_data()
    full = concat((train, dev))

    lb = LabelEncoder()
    lb.fit(train.cat.values)
    y_train = lb.transform(train.cat.values)
    dev_y = lb.transform(dev.cat.values)
    y_full = lb.transform(full.cat.values)

    test = load_nli_frame(nli_test_dataset_fn, nli_test_index_fn)

    folds_data = get_folds_data()

    for feature_set in feature_sets:
        if not feature_set_map.has_key(feature_set):
            logging.warn("Feature set id %s not found" % feature_set)
            continue

        feature_args = feature_set_map[feature_set]

        ex = FeaturePipeline(**feature_args)
        ex.fit(train)

        x_dev = ex.transform(dev)
        x_full = ex.transform(full)
        x_test = ex.transform(test)

        model = GridSearchCV(LinearSVC(), {'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]},
                     verbose=1, n_jobs=jobs)
        model.fit(x_full, y_full)

        params = model.best_params_
        logging.info("Using params %s scoring %s" % (params, model.best_score_))
        model = LinearSVC(**params)
        model.fit(x_full, y_full)

        predict(model, x_dev, dev, feature_set + '_dev.csv')
        predict(model, x_test, test, feature_set + '.csv')

        do_fixed_folds()
