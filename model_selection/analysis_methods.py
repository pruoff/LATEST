import numpy as np
import pandas as pd


def prepare_plot_of_kfold_cv(plot_tuples):
    """

    :param plot_tuples: dict: {
            'METHOD_NAME': {
                'full_method_name': Full name of method,
                'accs': list of accuracy values,
                'f1s': list of f1-score values,
                'runtime': list of runtime values
                }
            }

    :return: df_test_accs: dataframe with columns ['method', 'accuracy'],
            acc_ranks: list of full_method_names ordered by rank of average accuarcy,
            df_test_f1s: dataframe with columns ['method', 'f1-score'],
            f1_ranks: list of full_method_names ordered by rank of average f1-score
    """

    # get order
    dtype_acc = [('method', 'S25'), ('accuracy_mean', float)]
    dtype_f1 = [('method', 'S25'), ('f1_mean', float)]
    # for name in plot_tuples:
    accs = []
    f1s = []
    for name in plot_tuples:
        for acc in plot_tuples[name]['accs']:
            accs.append((name, acc))
        for f1 in plot_tuples[name]['f1s']:
            f1s.append((name, f1))

    acc_means_df = pd.DataFrame(
        accs, columns=['method', 'accuracy'])
    f1_means_df = pd.DataFrame(
        f1s, columns=['method', 'f1-score'])
    acc_means_df = acc_means_df.groupby('method').mean()
    f1_means_df = f1_means_df.groupby('method').mean()
    acc_means = []
    f1_means = []
    for i in range(len(plot_tuples)):
        acc_means.append((acc_means_df.index[i],
                          acc_means_df.iloc[i].values[0]))
        f1_means.append((f1_means_df.index[i],
                         f1_means_df.iloc[i].values[0]))
        i += 1
    acc_means = np.array(acc_means, dtype=dtype_acc)
    f1_means = np.array(f1_means, dtype=dtype_f1)
    ordered_accs = np.sort(acc_means, order=['accuracy_mean'])[::-1]
    ordered_f1s = np.sort(f1_means, order=['f1_mean'])[::-1]
    print('The ordered accuracies are: ', ordered_accs)
    print('The ordered f1-scores are: ', ordered_f1s)
    # ordered_accs = [(name, meanacc#1), (name, meanacc#2), ...]
    
    # the method names to plot
    method_plot_names = {
        'RF-Clf': 'RF',
        'LR': 'LR',
        'GaussianNB': 'GNB',
        'kNN': 'kNN',
        'SVM-lin': 'SVM',
        'GradientBDT': 'GBDT',
        'fastai': 'FCNN'
    }
    
    acc_ranks = []
    test_accs = []
    name_count = 0
    for (name, _) in ordered_accs:
        name = name.decode('utf-8')
        if name_count % 2 == 1:
            full_name = '\n{}'.format(name)
        else:
            full_name = name
        acc_ranks.append(full_name)
        for fold_number in range(len(plot_tuples[name]['accs'])):
            test_accs.append([method_plot_names[name], plot_tuples[name]['accs'][fold_number]])
        name_count += 1
    f1_ranks = []
    test_f1s = []
        
    for (name, _) in ordered_f1s:
        name = name.decode('utf-8')
        if name_count % 2 == 1:
            full_name = '\n{}'.format(name)
        else:
            full_name = name
        f1_ranks.append(full_name)
        for fold_number in range(len(plot_tuples[name]['f1s'])):
            test_f1s.append([method_plot_names[name], plot_tuples[name]['f1s'][fold_number]])
        name_count += 1
    df_test_accs = pd.DataFrame(test_accs, columns=['method', 'accuracy-mean'])
    df_test_f1s = pd.DataFrame(test_f1s, columns=['method', 'f1-score-mean'])
    return df_test_accs, acc_ranks, df_test_f1s, f1_ranks


def plot_f1s(df_test_f1s, naive_f1,
               store_path='plots/default_f1s.png'):
    """ Plot average f1-scores of all methods contained in df_test_f1s.

    :param df_test_f1s: dataframe with columns ['method', 'f1-score']
    :param naive_f1: accuracy of naive guesser
    :param store_path: path to export the plot to
    :return: -
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 4))
    ax1 = sns.pointplot(x='method', y='f1-score-mean', data=df_test_f1s, figsize=(3, 4))
    ax1.axhline(y=naive_f1, xmin=0.045, xmax=0.955, color='black')
    plt.savefig(store_path, dpi=150)
    plt.show()


def plot_accs(df_test_accs, naive_acc,
              store_path='plots/default_scikit_learn_accs.png'):
    """ Plot average accuracies of all ML methods contained in df_test_f1s.

    :param df_test_accs: dataframe with columns ['method', 'accuracy']
    :param naive_acc: accuracy of naive guesser
    :param store_path: path to export the plot to
    :return: -
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 4))
    ax1 = sns.pointplot(x='method', y='accuracy-mean', data=df_test_accs)
    ax1.axhline(y=naive_acc, xmin=0.045, xmax=0.955, color='black')
    plt.savefig(store_path, dpi=150)
    plt.show()
    return


def plot_training_times(plot_tuples, store_path='plots/default_runtimes.png'):
    """ Plot training times from plot_tuples and store plot at given path

    :param plot_tuples: dict containing [method]['traintime']
    :param store_path: path to export the plot to
    :return: -
    """
    import matplotlib.pyplot as plt
    bar_fig, bar_ax = plt.subplots()
    bar_ax.set_title('Runtimes for training in seconds')
    names = []
    times = []
    for name in plot_tuples:
        names.append(name)
        times.append(plot_tuples[name]['traintime'])
    # indent every second title on x axis
    for i, name in enumerate(names):
        if i % 2 != 0:
            names[i] = '\n' + name
    plt.bar(names, times)
    plt.yscale("log")
    for i, v in enumerate(times):
        if v == 0:
            bar_ax.text(i - 0.25, 1, str(int(v)), color='black', fontweight='bold')
        else:
            bar_ax.text(i - 0.25, v + v / 10, str(int(v)), color='black',
                        fontweight='bold')
    plt.savefig(store_path, dpi=150)
    plt.show()
    return


def plot_inference_times(plot_tuples,
                         store_path='plots/default_inference_times.png'):
    """ Plot inference times from plot_tuples and store plot at given path

    :param plot_tuples: dict containing [method]['infertime']
    :param store_path: path where to store the plot
    :return: -
    """
    import matplotlib.pyplot as plt
    bar_fig, bar_ax = plt.subplots()
    bar_ax.set_title('Inference times for training in seconds')
    names = []
    times = []
    for name in plot_tuples:
        names.append(name)
        times.append(plot_tuples[name]['infertime'])
    # indent every second title on x axis
    for i, name in enumerate(names):
        if i % 2 != 0:
            names[i] = '\n' + name
    plt.bar(names, times)
    plt.yscale("log")
    for i, v in enumerate(times):
        if v == 0:
            bar_ax.text(i - 0.25, 1, str(int(v)), color='black', fontweight='bold')
        else:
            bar_ax.text(i - 0.25, v + v / 10, str(int(v)), color='black',
                        fontweight='bold')
    plt.savefig(store_path, dpi=150)
    plt.show()
    return


def save_as_pickle(model, path='Pickle/default.sav'):
    """Saves parameter as pickle in given path"""
    import pickle
    pickle.dump(model, open(path, 'wb'))
    return


def load_from_pickle(path):
    """Loads pickle from path"""
    import pickle
    return pickle.load(open(path, 'rb'))


def _get_fold_rank_sums(methods, plot_tuples):
    """ Output the sum of the rank in which fold performed worst """
    rank_list = {}
    fold_sums = {}
    for num in range(10):
        fold_sums[num] = 0
    for method in methods:
        rank_list[method] = []
        i = 0
        for f1 in plot_tuples[method][2]:
            rank_list[method].append((f1, i))
            i += 1
        rank_list[method].sort(key=lambda x: x[0])
        j = 0
        for f1, index in rank_list[method]:
            fold_sums[index] += j
            j += 1
    return fold_sums


def plot_rank_sums(methods, plot_tuples, training_attributes, store_path):
    """ Plot the sum of the rank in which fold performed worst """
    import matplotlib.pyplot as plt
    fold_sums = _get_fold_rank_sums(methods, plot_tuples)
    bar_fig, bar_ax = plt.subplots()
    bar_ax.set_title('Sum of rank in worst performing over all methods fitted')
    folds = np.arange(training_attributes['n_folds'])
    rank_sums = []
    for index in range(len(training_attributes['train_test_indices'])):
        rank_sums.append(fold_sums[index])
    plt.bar(folds, rank_sums)
    for i, v in enumerate(rank_sums):
        if v == 0:
            bar_ax.text(i - 0.25, 1, str(int(v)), color='black', fontweight='bold')
        else:
            bar_ax.text(i - 0.25, v + .5, str(int(v)), color='black', fontweight='bold')
    plt.savefig(store_path, dpi=150)
    plt.show()


def prepare_plot_of_search_results(methods, occupant, path_of_searches, iter_count):
    """

    :param methods: list of ML methods applied
    :param occupant: occupant ID
    :param path_of_searches: path where pickles of randomized_search objects are stored
    :param iter_count: iteration count of randomized_searches
    :return: df: dataframe containing aggregated f1-scores,
            ordered_f1s: list of tuples (name, max_f1-score) ordered by max_f1-score
    """
    plot_tuples = {}
    random_searches = {}
    for method in methods:
        random_searches[method] = load_from_pickle(
            '{}/{}_random_searches_{}.sav'.format(path_of_searches, occupant, method))
        plot_tuples[method] = (
            method,
            [0],
            random_searches[method].cv_results_['mean_test_score'],
            np.sum(random_searches[method].cv_results_['mean_fit_time']),
            np.sum(random_searches[method].cv_results_['mean_score_time'])
        )
    # get order
    dtype_f1 = [('method', 'S25'), ('f1_max', float)]
    # for name in plot_tuples:
    f1s = []
    for name in plot_tuples:
        for f1 in plot_tuples[name][2]:
            f1s.append((name, f1))
    f1_maxs_df = pd.DataFrame(
        f1s, columns=['method', 'f1-score'])
    f1_maxs_df = f1_maxs_df.groupby('method').max()
    f1_maxs = []
    i = 0
    for _ in plot_tuples:
        f1_maxs.append((f1_maxs_df.index[i],
                        f1_maxs_df.iloc[i].values[0]))
        i += 1
    f1_maxs = np.array(f1_maxs, dtype=dtype_f1)
    ordered_f1s = np.sort(f1_maxs, order=['f1_max'])[::-1]
    print('The ordered f1-scores are: ', ordered_f1s)
    df = pd.DataFrame([], index=range(iter_count))
    
    # the method names to plot
    method_plot_names = {
        'RF-Clf_opt': 'RF',
        'LR_opt': 'LR',
        'GaussianNB_opt': 'GNB',
        'kNN_opt': 'kNN',
        'SVM_opt': 'SVM',
        'GradientBDT_opt': 'GBDT'
    }
        
    for (name, _) in ordered_f1s:
        name = name.decode('utf-8')
        df[method_plot_names[name]] = pd.DataFrame(plot_tuples[name][2], index=df.index)
    return df, ordered_f1s
