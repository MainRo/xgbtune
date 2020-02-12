import operator
import functools
import itertools
import numpy as np
import xgboost as xgb


def get_logger(enabled):
    if enabled:
        return lambda i: print(i)
    
    return lambda i: None


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


def tune_xgb_param(fit, train_set, parameters, tune_list, tune_params, round_count, loss_compare, log):
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

    log("best loss: {}".format(best_loss))
    for index, param in enumerate(tune_params):
        log("best {}: {}".format(param, best_config[index]))
        parameters[param] = best_config[index]

    return parameters


def tune_xgb_pass(fit, d_train, base_params, tune_params, round_count, loss_compare, log):
    log("computing best round...")
    evals_result = {}
    round_count, _ = fit(base_params, d_train, round_count)
    log("best round: {}".format(round_count))

    log("tuning max_depth and min_child_weight ...")
    max_depth_range = tune_params.get('max_depth') or list(range(5,9))
    min_child_weight_range = tune_params.get('min_child_weight') or list(range(1,4))
    tune_list = itertools.product(max_depth_range, min_child_weight_range)
    base_params = tune_xgb_param(
        fit,
        d_train, base_params,
        tune_list, ['max_depth', 'min_child_weight'],
        round_count, loss_compare, log)

    log("tuning gamma ...")
    gamma_range = tune_params.get('gamma') or [(i/10.0,) for i in range(0,5)]
    tune_list = gamma_range
    base_params = tune_xgb_param(
        fit,
        d_train, base_params,
        tune_list, ['gamma'],
        round_count, loss_compare, log)

    log("re-computing best round...")
    round_count, _ = fit(base_params, d_train, round_count)
    log("best round: {}".format(round_count))

    log("tuning subsample and colsample_bytree ...")
    subsample_range = tune_params.get('subsample') or [i/10.0 for i in range(6,11)]
    colsample_bytree_range = tune_params.get('colsample_bytree') or [i/10.0 for i in range(0,11)]
    tune_list = itertools.product(subsample_range, colsample_bytree_range)
    base_params = tune_xgb_param(
        fit,
        d_train, base_params,
        tune_list, ['subsample', 'colsample_bytree'],
        round_count, loss_compare, log)

    log("fine tuning subsample and colsample_bytree ...")
    subsample_range = tune_params.get('subsample') or [i/100 for i in range(
        int(base_params['subsample']*100-5),
        int(base_params['subsample']*100+5),
        5)]
    colsample_bytree_range = tune_params.get('colsample_bytree') or [i/100 for i in range(
        int(base_params['colsample_bytree']*100-5),
        int(base_params['colsample_bytree']*100+5),
        5
    )]

    tune_list = itertools.product(subsample_range, colsample_bytree_range)
    base_params = tune_xgb_param(
        fit,
        d_train, base_params,
        tune_list, ['subsample', 'colsample_bytree'],
        round_count, loss_compare, log)

    log("tuning alpha and lambda ...")
    alpha_range = tune_params.get('alpha') or [0, 0.01, 0.1, 0.5, 1]
    lambda_range = tune_params.get('lambda') or [1, 1.1, 1.5, 2, 5]

    tune_list = itertools.product(alpha_range, lambda_range)
    base_params = tune_xgb_param(
        fit,
        d_train, base_params,
        tune_list, ['alpha', 'lambda'],
        round_count, loss_compare, log)

    log("tuning seed ...")
    seed_range = tune_params.get('seed') or [(0,), (4,), (7,), (13,), (27,), (42,), (54,)]
    base_params = tune_xgb_param(
        fit,
        d_train, base_params,
        seed_range, ['seed'],
        round_count, loss_compare, log)

    log(base_params)
    return base_params, round_count


def tune_xgb_model(
    params, 
    x_train, y_train, x_val=None, y_val=None,
    nfold=3, stratified=False, folds=None, shuffle=True,
    tune_params={},
    max_round_count=5000, loss_compare=operator.lt, pass_count=2,
    verbose=True):
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
        tune_params: dictionary containing list of values to test to each parameter
        max_round_count: Maximum number of rounds during training
        pass_count: Number of tuning pass to do

    Returns:
        A tuple of tuned parameters and round count. 
    '''

    log = get_logger(verbose)
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
        log('tuning pass {}...'.format(tune_pass))
        params, round_count = tune_xgb_pass(
            fit,
            d_train,
            params,
            tune_params,
            round_count=max_round_count,
            loss_compare=loss_compare,
            log=log)

    return params, round_count
