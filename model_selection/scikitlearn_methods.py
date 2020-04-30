"""This is the scikitlearn module

It provides methods that run scikit-learn's classifiers on the provided data.
In particular, it covers a RandomizedSearchCV over pipelines consisting of a
StandardScaler and a following Random Forest / SVM / Gaussian Naive Bayes Classifier /
kNN / Gradient Boosting Desicion Tree / Linear Regressor.
It also gives methods to fit all classifiers
"""

__version__ = '0.1'
__author__ = 'Patrick Ruoff'

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import time as time
import numpy as np

import analysis_methods as ana


def run_random_search(methods, param_distributions,
                      n_iter, training_attributes, random_state):
    """ Run a RandomizedSearchCV on the data provided

    Use f1-score as metric.

    :param methods: array of method identifiers, e.g. 'LR' for Logistic Regression
    :param param_distributions: dict = {
                                    method = {
                                        ['clf__{}'.format(clf_parameter), ..]
                                    }
                                }
    :param n_iter: iteration count of randomized_search
    :param training_attributes: dict = {
        'occupant': occupant,
        'pos_label': pos_label,
        'X_df': X_df,
        'X_array': np.array(X_df),
        'Y_df': Y_df,
        'Y_array': np.array(Y_df),
        'train_test_indices': train_test_indices,
        'train_test_indices_int': Cross-Validation Split iterable with:
            "for train, test in train_test_indices_int:"
    }
    :param random_state: numpy random state to use for classifiers with random component

    :return: plot_tuples: dict = {
                method = {
                    'accs': [0],
                    'f1s': f1-scores,
                    'traintime': sum of fitting time,
                    'infertime': sum of inference time
                }
            },
            random_searches: dict = {
                method = RandomizedSearchCV object of corresponding method with
                    Pipeline consisting of StandardScaler and Classifier
                }
    """
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    from sklearn.metrics import make_scorer

    f1_scorer = make_scorer(f1_score, pos_label=training_attributes['pos_label'])
    plot_tuples = {}
    random_searches = {}
    long_name = ''
    for method in methods:
        if method == 'SVM_opt':
            from sklearn.svm import SVC
            estimators = [
                ('scale', StandardScaler()),
                ('clf', SVC(max_iter=5000, probability=True,
                            random_state=random_state))
            ]
            long_name = 'Optimized Support Vector Machine'
        elif method == 'RF-Clf_opt':
            from sklearn.ensemble import RandomForestClassifier
            estimators = [
                ('scale', StandardScaler()),
                ('clf', RandomForestClassifier(random_state=random_state))
            ]
            long_name = 'Optimized Random Forest Classifier'
        elif method == 'LR_opt':
            from sklearn.linear_model import LogisticRegression
            estimators = [
                ('scale', StandardScaler()),
                ('clf', LogisticRegression(solver='lbfgs', max_iter=5000,
                                           random_state=random_state))
            ]
            long_name = 'Optimized Logistic Regression'
        elif method == 'GaussianNB_opt':
            from sklearn.naive_bayes import GaussianNB
            estimators = [
                ('scale', StandardScaler()),
                ('clf', GaussianNB())
            ]
            long_name = 'Optimized Gaussian Naive Bayes'
        elif method == 'GradientBDT_opt':
            from sklearn.ensemble import GradientBoostingClassifier
            estimators = [
                ('scale', StandardScaler()),
                ('clf', GradientBoostingClassifier(
                    criterion='friedman_mse',
                    random_state=random_state))
            ]
            long_name = 'Optimized Gradient Boosting Desicion Trees'
        elif method == 'kNN_opt':
            from sklearn import neighbors
            estimators = [
                ('scale', StandardScaler()),
                ('clf', neighbors.KNeighborsClassifier())
            ]
            long_name = 'Optimized k-Nearest Neighbors'

        print(method)
        pipe = Pipeline(estimators)
        random_searches[method] = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_distributions[method],
            n_iter=n_iter, scoring=f1_scorer, n_jobs=-1,
            cv=training_attributes['train_test_indices_int'],
            refit=True, random_state=random_state, pre_dispatch='2*n_jobs',
            verbose=0
        )
        random_searches[method].fit(
            training_attributes['X_array'], training_attributes['Y_array']
        )
        plot_tuples[method] = {
            'long_name': long_name,
            'accs': [0],
            'random_searches': random_searches[method].cv_results_['mean_test_score'],
            'train_time': np.mean(random_searches[method].cv_results_['mean_fit_time']),
            'inference_time': np.mean(
                random_searches[method].cv_results_['mean_score_time'])
        }
        ana.save_as_pickle(
            random_searches[method],
            'Pickle/Other/{}_random_searches_{}.sav'.format(training_attributes['occupant'],
                                                            method)
        )
    return plot_tuples, random_searches


def fit_scikit_learn_methods(methods, training_attributes, rs, svm_linear_max_iter=5000,
                             rf_n_estimators=200):
    """ Fit the passed methods with scikit-learn's standard classifiers on passed data

    :param methods: array of method identifiers, e.g. 'LR' for Logistic Regression
    :param training_attributes: dict = {
        'occupant': occupant,
        'pos_label': pos_label,
        'X_df': X_df,
        'X_array': np.array(X_df),
        'Y_df': Y_df,
        'Y_array': np.array(Y_df),
        'train_test_indices': train_test_indices,
        'train_test_indices_int': Cross-Validation Split iterable with:
            "for train, test in train_test_indices_int:"
        )
    }
    :param rs: numpy random state to use for classifiers with random component
    :param svm_linear_max_iter: maximum number of iterations for SVM
    :param rf_n_estimators: number of estimators to use for Random Forest Clf and Reg
    :return: dict = {
                method = {
                    'accs': array of accuracies,
                    'f1s': array of f1-scores,
                    'traintime': sum of fitting time,
                    'infertime': sum of inference time
                }
            }
    """
    plot_tuples = {}
    for method in methods:
        if method == 'SVM-lin':
            plot_tuples[method] = _run_svm_linear(method, training_attributes, rs,
                                                  svm_linear_max_iter)
            plot_tuples[method]['long_name'] = 'Support Vector Machine with linear Kernel'
        elif method == 'RF-Clf':
            plot_tuples[method] = _run_rf_clf(method, training_attributes, rs,
                                              rf_n_estimators)
            plot_tuples[method]['long_name'] = 'Random Forest Classifier'
        elif method == 'RF-Reg':
            plot_tuples[method] = _run_rf_reg(method, training_attributes, rs,
                                              rf_n_estimators)
            plot_tuples[method]['long_name'] = 'Random Forest Regressor'
        elif method == 'LR':
            plot_tuples[method] = _run_lr(method, training_attributes, rs)
            plot_tuples[method]['long_name'] = 'Logistic Regression'
        elif method == 'GaussianNB':
            plot_tuples[method] = _run_gnb(method, training_attributes)
            plot_tuples[method]['long_name'] = 'Gaussian Naive Bayes'
        elif method == 'GradientBDT':
            plot_tuples[method] = _run_gbdt(method, training_attributes, rs)
            plot_tuples[method]['long_name'] = 'Gradient Boosting Decision Trees'
        elif method == 'kNN':
            plot_tuples[method] = _run_knn(method, training_attributes)
            plot_tuples[method]['long_name'] = 'k-Nearest Neighbors'
    return plot_tuples


def _fit_classifier(method, clf, training_attributes):
    """ Fit a sklearn classifier on the given data

    Save fitted model as pickle in 'Pickle/Models/{}.sav'.format(method)

    :param method: method identifier, e.g. 'RF-clf' for Random Forest Classifier
    :param clf: sklearn classifier to fit
    :param training_attributes: dict = {
        'occupant': occupant,
        'pos_label': pos_label,
        'X_df': X_df,
        'X_array': np.array(X_df),
        'Y_df': Y_df,
        'Y_array': np.array(Y_df),
        'train_test_indices': train_test_indices,
        'train_test_indices_int': Cross-Validation Split iterable with:
            "for train, test in train_test_indices_int:"
        )
    }

    :return: dict = {
                'accs': array of accuracies,
                'f1s': array of f1-scores,
                'traintime': sum of fitting time,
                'infertime': sum of inference time
            }
    """
    accs = []
    f1s = []
    train_time_sum = 0
    infer_time_sum = 0
    i = 0
    for train_index, test_index in training_attributes['train_test_indices']:
        print("Running fold {} of {}.".format(
            i, len(training_attributes['train_test_indices'])))
        train_time_start = time.time()
        clf = clf.fit(
            training_attributes['X_df'].loc[train_index],
            training_attributes['Y_df'].loc[train_index])
        train_time_sum += time.time() - train_time_start
        infer_time_start = time.time()
        # use around for regression output of Random Forest Regressor
        y_pred = np.around(
            clf.predict(training_attributes['X_df'].loc[test_index]))
        infer_time_sum += time.time() - infer_time_start
        accs.append(accuracy_score(
            training_attributes['Y_df'].loc[test_index], y_pred))
        f1s.append(f1_score(
            training_attributes['Y_df'].loc[test_index],
            y_pred,
            pos_label=training_attributes['pos_label'],
            average='binary'))
        i += 1
    print("Learning took", train_time_sum, "and Inference took",
          infer_time_sum, "seconds.")
    ana.save_as_pickle(clf, path='Pickle/Models/{}.sav'.format(method))
    return {
        'accs': accs,
        'f1s': f1s,
        'traintime': train_time_sum,
        'infertime': infer_time_sum
    }


# Support Vector Machine with linear Kernel
def _run_svm_linear(method, training_attributes, rs, max_iter):
    """ run a linear SVM classifier

    :param method: 'SVM-lin'
    :param training_attributes: dict = {
        'occupant': occupant,
        'pos_label': pos_label,
        'X_df': X_df,
        'X_array': np.array(X_df),
        'Y_df': Y_df,
        'Y_array': np.array(Y_df),
        'train_test_indices': train_test_indices,
        'train_test_indices_int': Cross-Validation Split iterable with:
            "for train, test in train_test_indices_int:"
    }
    :param rs: the numpy random state
    :param max_iter: the maximum number of iterations for the SVM Classifier
    :return: dict = {
                'accs': array of accuracies,
                'f1s': array of f1-scores,
                'traintime': sum of fitting time,
                'infertime': sum of inference time
            }
    """

    from sklearn.svm import SVC

    print('Running SVM with linear kernel and '
          '{}-Fold-Cross Validation..'.format(
        str(len(training_attributes['train_test_indices']))))
    clf = SVC(max_iter=max_iter, random_state=rs)
    return _fit_classifier(method, clf, training_attributes)


# Random Forest Classifier
def _run_rf_clf(method, training_attributes, rs, n_estimators=200):
    """ run a linear SVM classifier

    :param method: 'RF-Clf'
    :param training_attributes: dict = {
        'occupant': occupant,
        'pos_label': pos_label,
        'X_df': X_df,
        'X_array': np.array(X_df),
        'Y_df': Y_df,
        'Y_array': np.array(Y_df),
        'train_test_indices': train_test_indices,
        'train_test_indices_int': Cross-Validation Split iterable with:
            "for train, test in train_test_indices_int:"
    }
    :param rs: the numpy random state
    :param n_estimators: the number of estimators for the Random Forest Classifier
    :return: dict = {
                'accs': array of accuracies,
                'f1s': array of f1-scores,
                'traintime': sum of fitting time,
                'infertime': sum of inference time
            }
    """

    from sklearn.ensemble import RandomForestClassifier

    print('Running Random Forest Classifier with '
          '{}-Fold-Cross Validation..'.format(
        str(len(training_attributes['train_test_indices']))))
    clf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=rs)
    return _fit_classifier(method, clf, training_attributes)


# Random Forest Regressor
def _run_rf_reg(method, training_attributes, rs, n_estimators=200):
    """ run a Random Forest Regressor on the classification problem

    :param method: 'RF-Reg'
    :param training_attributes: dict = {
        'occupant': occupant,
        'pos_label': pos_label,
        'X_df': X_df,
        'X_array': np.array(X_df),
        'Y_df': Y_df,
        'Y_array': np.array(Y_df),
        'train_test_indices': train_test_indices,
        'train_test_indices_int': Cross-Validation Split iterable with:
            "for train, test in train_test_indices_int:"
    }
    :param rs: the numpy random state
    :param n_estimators: the number of estimators for the Random Forest Regressor
    :return: dict = {
                'accs': array of accuracies,
                'f1s': array of f1-scores,
                'traintime': sum of fitting time,
                'infertime': sum of inference time
            }
    """
    from sklearn.ensemble import RandomForestRegressor
    print('Running Random Forest Regressor with '
          '{}-Fold-Cross Validation..'.format(
        str(len(training_attributes['train_test_indices']))))
    clf = RandomForestRegressor(
        n_estimators=n_estimators, random_state=rs)
    return _fit_classifier(method, clf, training_attributes)


# Logistic Regression
def _run_lr(method, training_attributes, rs, max_iter=200):
    """ run a Linear Regression Classifier

    :param method: 'LR'
    :param training_attributes: dict = {
        'occupant': occupant,
        'pos_label': pos_label,
        'X_df': X_df,
        'X_array': np.array(X_df),
        'Y_df': Y_df,
        'Y_array': np.array(Y_df),
        'train_test_indices': train_test_indices,
        'train_test_indices_int': Cross-Validation Split iterable with:
            "for train, test in train_test_indices_int:"
    }
    :param rs: the numpy random state
    :param max_iter: the number of maximum iterations for the Linear Regression Classifier
    :return: dict = {
                'accs': array of accuracies,
                'f1s': array of f1-scores,
                'traintime': sum of fitting time,
                'infertime': sum of inference time
            }
    """
    from sklearn.linear_model import LogisticRegression

    print('Running Logistic Regression with '
          '{}-Fold-Cross Validation..'.format(
        str(len(training_attributes['train_test_indices']))))

    clf = LogisticRegression(solver='lbfgs', max_iter=max_iter,
                             random_state=rs)
    return _fit_classifier(method, clf, training_attributes)

# Gaussian Naive Bayes
def _run_gnb(method, training_attributes):
    """ run a Gaussian Naive Bayes Classifier

    :param method: 'GNB'
    :param training_attributes: dict = {
        'occupant': occupant,
        'pos_label': pos_label,
        'X_df': X_df,
        'X_array': np.array(X_df),
        'Y_df': Y_df,
        'Y_array': np.array(Y_df),
        'train_test_indices': train_test_indices,
        'train_test_indices_int': Cross-Validation Split iterable with:
            "for train, test in train_test_indices_int:"
    }
    :return: dict = {
                'accs': array of accuracies,
                'f1s': array of f1-scores,
                'traintime': sum of fitting time,
                'infertime': sum of inference time
            }
    """

    from sklearn.naive_bayes import GaussianNB

    print('Running Gaussian Naive Bayes with '
          '{}-Fold-Cross Validation..'.format(
        str(len(training_attributes['train_test_indices']))))

    clf = GaussianNB()

    return _fit_classifier(method, clf, training_attributes)


# Gradient Boosting Desicion Trees
def _run_gbdt(method, training_attributes, rs):
    """ run a Gradient Boosting Decision Tree

    :param method: 'GBDT'
    :param training_attributes: dict = {
        'occupant': occupant,
        'pos_label': pos_label,
        'X_df': X_df,
        'X_array': np.array(X_df),
        'Y_df': Y_df,
        'Y_array': np.array(Y_df),
        'train_test_indices': train_test_indices,
        'train_test_indices_int': Cross-Validation Split iterable with:
            "for train, test in train_test_indices_int:"
    }
    :param rs: the numpy random state
    :return: dict = {
                'accs': array of accuracies,
                'f1s': array of f1-scores,
                'traintime': sum of fitting time,
                'infertime': sum of inference time
            }
    """

    from sklearn.ensemble import GradientBoostingClassifier

    print('Running Gradient Boosting Desicion Trees with '
          '{}-Fold-Cross Validation..'.format(
        str(len(training_attributes['train_test_indices']))))

    clf = GradientBoostingClassifier(random_state=rs)

    return _fit_classifier(method, clf, training_attributes)


# k-Nearest Neighbors Classifier
def _run_knn(method, training_attributes, n_neighbors=10):
    """ run a k-Nearest Neighbors Classifier

    :param method: 'kNN'
    :param training_attributes: dict = {
        'occupant': occupant,
        'pos_label': pos_label,
        'X_df': X_df,
        'X_array': np.array(X_df),
        'Y_df': Y_df,
        'Y_array': np.array(Y_df),
        'train_test_indices': train_test_indices,
        'train_test_indices_int': Cross-Validation Split iterable with:
            "for train, test in train_test_indices_int:"
    }
    :param rs: the numpy random state
    :return: dict = {
                'accs': array of accuracies,
                'f1s': array of f1-scores,
                'traintime': sum of fitting time,
                'infertime': sum of inference time
            }
    """

    from sklearn import neighbors

    print('Running k-Nearest Neighbors with Statified-'
          '{}-Fold-Cross Validation..'.format(
        str(len(training_attributes['train_test_indices']))))
    clf = neighbors.KNeighborsClassifier(n_neighbors)

    return _fit_classifier(method, clf, training_attributes)
