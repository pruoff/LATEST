import pandas as pd
import numpy as np


def import_data(occupant):
    """ Import data for one occupant

    :param occupant: ID of occupant to import data for
    :return: dataframe of occupant's data with datetimeindex indices
    """
    path = '~/ma-patrick-ruoff/09 Data/{}_preprocessedData.csv'.format(
        occupant)
    data = pd.read_csv(path)
    data.index = data['index'].values
    data = data.drop(columns=['index'])
    data.index = pd.to_datetime(data.index)
    data.index = data.index.rename('timestamp')
    data.index = data.index.tz_convert(None)
    return data


def prepare_data(data, target):
    """ Prepare data by calculating naive accuracy and f1-score, the positive label,
    and separating the data into X and Y by target

    :param data: dataframe of occupant's data to prepare
    :param target: str of target feature
    :return:
        'random_acc': accuracy of naive guesser,
        'random_f1': f1-score of naive guesser,
        'positive_label': label defined to be positive for f1-score calculation
        'Y_df': dataframe of target data,
        'X_df': dataframe of training data
    """
    print('The target variable distribution is:\n',
          data[target].value_counts() / data.shape[0])
    random_acc = 1 - (data[target].value_counts() / data.shape[0])[1]
    random_f1 = 2 * random_acc / (random_acc + 1)
    positive_label = int(data[target].value_counts().index[0])
    # convert to np.ndarray
    Y_df = data[target]
    X_df = data.drop(columns=[target])
    # one-hot encode all categorical features
    X_df = pd.get_dummies(X_df)
    print('X shape:', np.shape(X_df))
    return random_acc, random_f1, positive_label, Y_df, X_df


def print_missing_entries(data):
    """ Print missing data points to console if any

    :param data: dataframe
    :return: -
    """
    missing_columns = data.columns[data.isna().any().values]
    for col in missing_columns:
        print(col)
        print(data[col][data[col].isna()])
    return


def _get_consecutive_intervals(data, target):
    """ return: array of tuples with:
    (interval_array, #datapoints, #pos_labels)
    """
    is_very_first = True
    interval_count = 0
    for current_index in data.index:
        if is_very_first:
            intervals = np.array(
                [(current_index, 0, 0)],
                dtype=[('indices', np.ndarray),
                       ('#datapoints', 'int'),
                       ('#pos_labels', 'int')])
            is_very_first = False
        elif not last_index + pd.Timedelta('30 seconds') == current_index:
            # new first index reached
            intervals = np.concatenate(
                (intervals, [np.array((current_index, 0, 0),
                                      dtype=[('indices', np.ndarray),
                                             ('#datapoints', 'int'),
                                             ('#pos_labels', 'int')])]))
            interval_count += 1
        # increment counts
        intervals[interval_count]['#datapoints'] += 1
        if data[target].loc[current_index] == 1:
            intervals[interval_count]['#pos_labels'] += 1
        last_index = current_index
    return intervals


def interval_stratified_k_fold_cross_validation(
        data, n_folds, target, random_state, seed):
    """ Split intervals into n_folds folds
    Even out label distribution by putting new interval in smallest fold

    return: iterable with train, test indices as pd.DatetimeIndex
    """
    intervals = _get_consecutive_intervals(data, target)
    # numpy requires the seed to be set every time right before permuation
    random_state.seed(seed)
    pos_interval_count = 0
    for interval in intervals:
        if interval[2]:
            pos_interval_count += 1
    if pos_interval_count < n_folds:
        print('Given number of folds is smaller than available '
              'time intervals containing a positive true label!'
              'Reducing number of folds to {}'.format(pos_interval_count))
        n_folds = pos_interval_count
    fold_quantities = np.array(
        [(0, 0, 0)],
        dtype=[('index', int), ('#datapoints', 'int'), ('#pos_labels', 'int')]
    )
    for i in np.arange(n_folds - 1) + 1:
        fold_quantities = np.append(fold_quantities, np.array(
            [(i, 0, 0)],
            dtype=[('index', int), ('#datapoints', 'int'), ('#pos_labels', 'int')]
        ))
    train_test_indices = []
    for i in range(n_folds):
        train_test_indices.append([data.index.copy(), []])

    pos_count_ordered = sorted(intervals, key=lambda interval: interval[2])[::-1]
    for interval in pos_count_ordered:
        # check if interval contains pos label
        if interval[2]:
            insert_index = sorted(fold_quantities, key=lambda x: x['#pos_labels'])[0][0]
        else:
            insert_index = sorted(fold_quantities, key=lambda x: x['#datapoints'])[0][0]
        # insert into fold
        interval_range = pd.date_range(
            start=interval[0], periods=interval[1], freq='30S')
        start_index_in_data = np.where(train_test_indices[insert_index][0] == interval[0])
        # update train and test indices
        for index in interval_range:
            train_test_indices[insert_index][1].append(index)
            train_test_indices[insert_index][0] = np.delete(
                train_test_indices[insert_index][0], start_index_in_data)
        fold_quantities[insert_index]['#datapoints'] += interval['#datapoints']
        fold_quantities[insert_index]['#pos_labels'] += interval['#pos_labels']

    print('Training and validation set sizes are:')
    fold = 0
    for train, test in train_test_indices:
        print(' Fold {}:   train/valid  {}/{}   with {} positive labels'.format(
            fold, np.shape(train)[0], np.shape(test)[0],
            fold_quantities[fold]['#pos_labels']))
        fold += 1
    return train_test_indices


def transform_date_time_index_to_int(data, train_test_indices):
    """Transforms the indices in date time index format to int

    :param data: dataframe of data
    :param train_test_indices: dict of train and test indices as date time index
    :return: dict of train an test indices as int
    """
    index_dict = {}
    for datetime_index in data.index:
        index_dict[datetime_index] = np.where(
            data.index == datetime_index)[0][0]

    train_test_indices_int = []
    fold_count = 0
    for train, test in train_test_indices:
        train_test_indices_int.append([[], []])
        for index in train:
            train_test_indices_int[fold_count][0].append(
                index_dict[index])
        for index in test:
            train_test_indices_int[fold_count][1].append(
                index_dict[index])
        fold_count += 1
    return train_test_indices_int
