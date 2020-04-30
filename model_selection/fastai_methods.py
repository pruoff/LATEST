import fastai
import fastai.tabular
import numpy as np
import time


def run_k_fold_cross_validation_fastai(data, training_attributes, epochs,
                           learning_rate, cat_names, cont_names, target,
                           procs, layers):
    """ Execute a k-fold cross-validation with fastai on the given data.
    
    :param data: full dataframe of all data without X/Y separation
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
    :param epochs: number of epochs as int
    :param learning_rate: learning rate for training the NN
    :param cat_names: categorical feature names 
    :param cont_names: continuous feature names
    :param procs: list of fastai processing steps 
    :param layers: list of layers containing node_counts as int
    :return: dict = {
                'train_accs': array of training accuracies,
                'accs': array of accuracies,
                'train_f1s': array of training f1-scores,
                'f1s': array of f1-scores,
                'traintime': sum of fitting time,
                'infertime': sum of inference time
            }
    """

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score

    train_accs = []
    val_accs = []
    train_f1s = []
    val_f1s = []
    train_time_sum = 0
    infer_time_sum = 0
    
    # error will be thrown otherwise
    data = data.astype({target: int})
    
    # use new learner for every fold
    for train_index, val_index in training_attributes['train_test_indices_int']:
        fast_data = fastai.tabular.TabularDataBunch.from_df(
            path='.', df=data, dep_var=target,
            valid_idx=val_index, procs=procs, 
            # batch size for correct prediction shape
            bs=np.shape(train_index)[0],
            cat_names=cat_names, cont_names=cont_names)
        learn = fastai.tabular.tabular_learner(
            fast_data, layers=layers, metrics=fastai.tabular.accuracy)
  
        train_time_start = time.time()
        learn.fit_one_cycle(epochs, learning_rate)
        train_time_sum += time.time() - train_time_start

        pos_label_in_learner = _get_pos_label_in_learner(
            learn, training_attributes['Y_df'], 
            training_attributes['train_test_indices_int'],
            training_attributes['pos_label'])
        
        infer_time_start = time.time()

        # score
        train_pred = learn.validate(learn.data.train_dl)
        test_pred = learn.validate(learn.data.valid_dl)

        test_preds, y_true_test = learn.get_preds(
            ds_type=fastai.tabular.DatasetType.Valid, with_loss=False)
        train_preds, y_true_train = learn.get_preds(
            ds_type=fastai.tabular.DatasetType.Train, with_loss=False)
        infer_time_sum += time.time() - infer_time_start

        y_true_test = fastai.tabular.to_np(y_true_test)
        y_true_train = fastai.tabular.to_np(y_true_train)
        
        # document predictions in correct format
        test_y_preds = fastai.tabular.to_np(test_preds.argmax(dim=-1))
        train_y_preds = fastai.tabular.to_np(train_preds.argmax(dim=-1))
        val_accs.append(accuracy_score(y_true_test, test_y_preds))
        val_f1s.append(f1_score(
            y_true_test, test_y_preds, average='binary',
            pos_label=pos_label_in_learner))
        train_accs.append(accuracy_score(
            y_true_train, train_y_preds))
        train_f1s.append(f1_score(
            y_true_train, train_y_preds, average='binary',
            pos_label=pos_label_in_learner))
    return {'train_accs': train_accs,
            'accs': val_accs,
            'train_f1s': train_f1s,
            'f1s': val_f1s,
            'traintime': train_time_sum,
            'infertime': infer_time_sum}


def _get_pos_label_in_learner(learn, Y_df, train_test_indices_int, pos_label):
    """ For f1-score: return pos_label that is set by fastai in learner. """
    pos_label_in_learner = 1
    _, y_true_in_model = learn.get_preds(
        ds_type=fastai.tabular.DatasetType.Valid, with_loss=False)
    for train_index, test_index in train_test_indices_int:
        y_true = Y_df.iloc[test_index][0]
        if y_true == pos_label:
            y_true = 1
        else:
            y_true = 0
        y_true_in_model = y_true_in_model.data
        y_true_in_model = fastai.tabular.to_np(y_true_in_model)
        if y_true_in_model[0] != y_true:
            # pos_label != pos_label_in_model so turn to 0
            pos_label_in_learner = 0
        break
    return pos_label_in_learner

