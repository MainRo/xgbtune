import operator
import functools
import itertools
import numpy as np
import xgboost as xgb


def _fit_val(xgb_fit, parameters, train_set, round_count):
    evals_result = {}
    model = xgb_fit(
        parameters, train_set,
        num_boost_round=round_count,
        early_stopping_rounds=10,
        evals_result=evals_result,
        verbose_eval=False,
    )

    loss = evals_result['validation'][parameters['eval_metric']][-1]
    best_round = model.best_iteration + 1
    return best_round, loss


def _fit_cv(xgb_fit, parameters, train_set, round_count):
    eval = xgb_fit(
        parameters, train_set,
        num_boost_round=round_count,
        early_stopping_rounds=10,
        verbose_eval=False,
    )

    best_round = eval.shape[0]
    loss = np.round(eval.iloc[-1, 0], 4)
    return best_round, loss


def tune_xgb_param(fit, train_set, parameters, tune_list, tune_params, round_count, loss_compare):
    best_loss = None
    best_config = None
    for tune_config in tune_list:
        for index, param in enumerate(tune_params):
            parameters[param] = tune_config[index]
        evals_result = {}
        _, loss = fit(parameters, train_set, round_count)

        if best_loss is None or loss_compare(loss, best_loss):
            best_loss = loss
            best_config = tune_config

    print("best loss: {}".format(best_loss))
    for index, param in enumerate(tune_params):
        print("best {}: {}".format(param, best_config[index]))
        parameters[param] = best_config[index]

    return parameters


def tune_xgb_pass(fit, d_train, base_params, round_count, loss_compare):
    print("computing best round...")
    evals_result = {}
    round_count, _ = fit(base_params, d_train, round_count)
    print("best round: {}".format(round_count))

    print("tuning max_depth and min_child_weight ...")
    max_depth_range = list(range(5,9))
    min_child_weight_range = list(range(1,4))
    tune_list = itertools.product(max_depth_range, min_child_weight_range)
    base_params = tune_xgb_param(
        fit,
        d_train, base_params,
        tune_list, ['max_depth', 'min_child_weight'], round_count, loss_compare)

    print("tuning gamma ...")
    gamma_range = [(i/10.0,) for i in range(0,5)]
    tune_list = gamma_range
    base_params = tune_xgb_param(
        fit,
        d_train, base_params,
        tune_list, ['gamma'], round_count, loss_compare)

    print("re-computing best round...")
    round_count, _ = fit(base_params, d_train, round_count)
    print("best round: {}".format(round_count))

    print("tuning subsample and colsample_bytree ...")
    subsample_range = [i/10.0 for i in range(6,11)]
    colsample_bytree_range = [i/10.0 for i in range(0,11)]
    tune_list = itertools.product(subsample_range, colsample_bytree_range)
    base_params = tune_xgb_param(
        fit,
        d_train, base_params,
        tune_list, ['subsample', 'colsample_bytree'], round_count, loss_compare)

    print("fine tuning subsample and colsample_bytree ...")
    subsample_range = [i/100 for i in range(
        int(base_params['subsample']*100-5),
        int(base_params['subsample']*100+5),
        5)]
    colsample_bytree_range = [i/100 for i in range(
        int(base_params['colsample_bytree']*100-5),
        int(base_params['colsample_bytree']*100+5),
        5
    )]

    tune_list = itertools.product(subsample_range, colsample_bytree_range)
    base_params = tune_xgb_param(
        fit,
        d_train, base_params,
        tune_list, ['subsample', 'colsample_bytree'],
        round_count, loss_compare)

    print("tuning alpha and lambda ...")
    alpha_range = [0, 0.01, 0.1, 0.5, 1]
    lambda_range = [1, 1.1, 1.5, 2, 5]

    tune_list = itertools.product(alpha_range, lambda_range)
    base_params = tune_xgb_param(
        fit,
        d_train, base_params,
        tune_list, ['alpha', 'lambda'],
        round_count, loss_compare)

    print("tuning seed ...")
    seed_range = [(0,), (4,), (7,), (13,), (27,), (42,), (54,)]
    base_params = tune_xgb_param(
        fit,
        d_train, base_params,
        seed_range, ['seed'], round_count, loss_compare)

    print(base_params)
    return base_params, round_count


def tune_xgb_model(
    params, 
    x_train, y_train, x_val=None, y_val=None,
    nfold=3, stratified=False, folds=None, shuffle=True,
    max_round_count=5000, loss_compare=operator.lt, pass_count=2):
    '''Tunes a XGBoost model

    Examples:
        >>> params, round_count = tune_xgb_model(x, y, x_val, y_val, model_params)

    Args:
        params: A dictionary with the base xgboost parameters to use
        x_train: Train set
        y_train: Train labels
        x_val: Validation set
        y_val: Validation labels
        nfold: Number of folds for cv
        stratified: Perform stratified sampling
        folds: Sklearn KFolds or StratifiedKFolds object
        shuffle: shuffle data on cross validation
        max_round_count: Maximum number of rounds during training
        pass_count: Number of tuning pass to do

    Returns:
        A tuple of tuned parameters and round count. 
    '''

    d_train = xgb.DMatrix(x_train, label=y_train)
    if x_val is None:  # cv
        kwargs = {
            'nfold': nfold,
            'stratified': stratified,
            'folds': folds,
            'shuffle': shuffle,
        }
        xgb_fit = functools.partial(xgb.cv, **kwargs)
        fit = functools.partial(_fit_cv, xgb_fit)
    else:  # validation set
        d_val = xgb.DMatrix(x_val, label=y_val)
        kwargs = {
            "evals": [(d_val, "validation")],
        }

        xgb_fit = functools.partial(xgb.train, **kwargs)
        fit = functools.partial(_fit_val, xgb_fit)
    

    for tune_pass in range(pass_count):
        print('tuning pass {}...'.format(tune_pass))
        params, round_count = tune_xgb_pass(
            fit,
            d_train,
            params, 
            round_count=max_round_count,
            loss_compare=loss_compare)

    return params, round_count
