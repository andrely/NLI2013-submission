import csv
import logging
from numpy import mean, std
from pandas import concat, pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from data import load_nli_data, load_nli_frame, nli_test_dataset_fn, nli_test_index_fn, folds_fn
from features import DEFAULT_FEATURES, TOKEN_COLLOCATION_FEATURE_ID, FeaturePipeline, SUFFIX_COLLOCATION_FEATURE_ID, extract_suffixes
from ten_fold import get_folds_data

jobs = 10

def predict(model, input, input_frame, out_fn):
    out = model.predict(input)
    out_label = lb.inverse_transform(out.astype(int))
    logging.info("Writing predictions to %s" % out_fn)
    with open(out_fn, 'w') as f:
        csv_out = csv.writer(f)

        for i, label in enumerate(out_label):
            csv_out.writerow([input_frame.file.ix[i], label])

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    train, dev = load_nli_data()
    full = concat((train, dev))

    lb = LabelEncoder()
    lb.fit(train.cat.values)
    y_train = lb.transform(train.cat.values)
    dev_y = lb.transform(dev.cat.values)
    y_full = lb.transform(full.cat.values)

    test = load_nli_frame(nli_test_dataset_fn, nli_test_index_fn)

    folds_data = get_folds_data()

    # Best system
    feature_args = {'features': DEFAULT_FEATURES + [TOKEN_COLLOCATION_FEATURE_ID, SUFFIX_COLLOCATION_FEATURE_ID],
                    'token_coll_args': {'window': 1,
                                        'directional': True},
                    'char_vect_args': {'ngram_range': (3, 6)},
                    'suff_coll_args': {'preprocessor': lambda text: extract_suffixes(text, suff_len=4),
                                       'directional': True,
                                       'window': 1}}

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
    # INFO:root:Using params {'C': 0.3} scoring 0.817181352308
     #INFO:root:Using params {'C': 1} scoring 0.827090137568
    model = LinearSVC(**params)
    model.fit(x_full, y_full)

    predict(model, x_dev, dev, 'full_svm_dev.csv')
    predict(model, x_test, test, 'full_svm.csv')

    logging.info("doing 10-fold")

    scores = []

    for i, (tf_train, tf_test) in enumerate(get_folds_data()):
        fold = i + 1
        logging.info("Fold %d" % i)

        ex = FeaturePipeline(**feature_args)
        tf_x_train = ex.fit_transform(tf_train)
        tf_y_train = lb.transform(tf_train.cat.values)

        tf_x_test = ex.transform(tf_test)
        tf_y_test = lb.transform(tf_test.cat.values)

        model = GridSearchCV(LinearSVC(), {'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]},
                             verbose=1, n_jobs=jobs)
        model.fit(tf_x_train, tf_y_test)

        out = model.predict(input)

        score = accuracy_score(tf_y_test, out)
        logging.info("Score %s" % score)
        scores.append(score)

    print "System 1 10-fold mean: %f, stddev: %f" % (mean(scores), std(scores))

    x_test = x_dev= x_full = model = tf_train = tf_x_train = tf_y_train, tf_test, tf_x_test, tf_y_test =  None

    # min_df 5
    feature_args = {'features': DEFAULT_FEATURES + [TOKEN_COLLOCATION_FEATURE_ID, SUFFIX_COLLOCATION_FEATURE_ID],
                    'token_vect_args': {'min_df': 5},
                    'char_vect_args': {'min_df': 5, 'ngram_range': (3, 6)},
                    'token_coll_args': {'min_df': 5,
                                        'window': 1,
                                        'directional': True},
                    'suff_coll_args': {'min_df': 5,
                                       'preprocessor': lambda text: extract_suffixes(text, suff_len=4),
                                       'directional': True,
                                       'window': 1}}

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
    # INFO:root:Using params {'C': 0.3} scoring 0.824272162333

    model = LinearSVC(**params)
    model.fit(x_full, y_full)

    predict(model, x_dev, dev, 'min_5_svm_dev.csv')
    predict(model, x_test, test, 'min_5_svm.csv')

    logging.info("doing 10-fold")

    scores = []

    for i, (tf_train, tf_test) in enumerate(get_folds_data()):
        fold = i + 1
        logging.info("Fold %d" % i)

        ex = FeaturePipeline(**feature_args)

        tf_x_train = ex.fit_transform(tf_train)
        tf_y_train = lb.transform(tf_train.cat.values)

        tf_x_test = ex.transform(tf_test)
        tf_y_test = lb.transform(tf_test.cat.values)

        model = GridSearchCV(LinearSVC(), {'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]},
                             verbose=1, n_jobs=jobs)
        model.fit(tf_x_train, tf_y_test)

        out = model.predict(input)

        score = accuracy_score(tf_y_test, out)
        logging.info("Score %s" % score)
        scores.append(score)

    print "System 2 10-fold mean: %f, stddev: %f" % (mean(scores), std(scores))

    x_test = x_dev= x_full = model = tf_train = tf_x_train = tf_y_train, tf_test, tf_x_test, tf_y_test =  None

    # min_df 10
    feature_args = {'features': DEFAULT_FEATURES + [TOKEN_COLLOCATION_FEATURE_ID, SUFFIX_COLLOCATION_FEATURE_ID],
                    'token_vect_args': {'min_df': 10},
                    'char_vect_args': {'min_df': 10,
                                       'ngram_range': (3, 6)},
                    'token_coll_args': {'min_df': 10,
                                        'window': 1,
                                        'directional': True},
                    'suff_coll_args': {'min_df': 10,
                                       'preprocessor': lambda text: extract_suffixes(text, suff_len=4),
                                       'directional': True,
                                       'window': 1}}

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
    # INFO:root:Using params {'C': 0.3} scoring 0.822726674701

    model = LinearSVC(**params)
    model.fit(x_full, y_full)

    predict(model, x_dev, dev, 'min_10_svm_dev.csv')
    predict(model, x_test, test, 'min_10_svm.csv')

    logging.info("doing 10-fold")

    scores = []

    for i, (tf_train, tf_test) in enumerate(get_folds_data()):
        fold = i + 1
        logging.info("Fold %d" % i)

        ex = FeaturePipeline(**feature_args)

        tf_x_train = ex.fit_transform(tf_train)
        tf_y_train = lb.transform(tf_train.cat.values)

        tf_x_test = ex.transform(tf_test)
        tf_y_test = lb.transform(tf_test.cat.values)

        model = GridSearchCV(LinearSVC(), {'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]},
                             verbose=1, n_jobs=jobs)
        model.fit(tf_x_train, tf_y_test)

        out = model.predict(input)

        score = accuracy_score(tf_y_test, out)
        logging.info("Score %s" % score)
        scores.append(score)

    print "System 3 10-fold mean: %f, stddev: %f" % (mean(scores), std(scores))

    x_test = x_dev= x_full = model = tf_train = tf_x_train = tf_y_train, tf_test, tf_x_test, tf_y_test =  None

    # mixed
    feature_args = {'features': DEFAULT_FEATURES + [TOKEN_COLLOCATION_FEATURE_ID, SUFFIX_COLLOCATION_FEATURE_ID],
                    'token_vect_args': {'min_df': 10},
                    'char_vect_args': {'min_df': 10,
                                       'ngram_range': (1, 7)},
                    'token_coll_args': {'min_df': 5,
                                        'window': 1,
                                        'directional': True},
                    'suff_coll_args': {'min_df': 5,
                                       'preprocessor': lambda text: extract_suffixes(text, suff_len=4),
                                       'directional': True,
                                       'window': 1}}

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
    # INFO:root:Using params {'C': 0.3} scoring 0.823635782156

    model = LinearSVC(**params)
    model.fit(x_full, y_full)

    predict(model, x_dev, dev, 'min_mixed_svm_dev.csv')
    predict(model, x_test, test, 'min_mixed_svm.csv')

    logging.info("doing 10-fold")

    scores = []

    for i, (tf_train, tf_test) in enumerate(get_folds_data()):
        fold = i + 1
        logging.info("Fold %d" % i)

        ex = FeaturePipeline(**feature_args)

        tf_x_train = ex.fit_transform(tf_train)
        tf_y_train = lb.transform(tf_train.cat.values)

        tf_x_test = ex.transform(tf_test)
        tf_y_test = lb.transform(tf_test.cat.values)

        model = GridSearchCV(LinearSVC(), {'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]},
                             verbose=1, n_jobs=jobs)
        model.fit(tf_x_train, tf_y_test)

        out = model.predict(input)

        score = accuracy_score(tf_y_test, out)
        logging.info("Score %s" % score)
        scores.append(score)

    print "System 4 10-fold mean: %f, stddev: %f" % (mean(scores), std(scores))

    x_test = x_dev= x_full = model = tf_train = tf_x_train = tf_y_train, tf_test, tf_x_test, tf_y_test =  None
