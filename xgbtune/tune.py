import operator
import itertools
import xgboost as xgb


def tune_xgb_param(model, train_set, val_set, parameters, tune_list, tune_params, round_count, loss_compare):
    best_loss = None
    best_config = None
    for tune_config in tune_list:
        for index, param in enumerate(tune_params):
            parameters[param] = tune_config[index]
        evals_result = {}
        xgb.train(
            parameters, train_set,
            num_boost_round=round_count,
            early_stopping_rounds=10,
            evals=[(val_set, "validation")],
            evals_result=evals_result,
            verbose_eval=False,
        )

        loss = evals_result['validation'][parameters['eval_metric']][-1]
        if best_loss is None or loss_compare(loss, best_loss):
            best_loss = loss
            best_config = tune_config

    print("best loss: {}".format(best_loss))
    for index, param in enumerate(tune_params):
        print("best {}: {}".format(param, best_config[index]))
        parameters[param] = best_config[index]

    return parameters


def tune_xgb_pass(train, target, val, val_target, base_params, round_count, loss_compare):
    d_train = xgb.DMatrix(train, label=target)
    d_val = xgb.DMatrix(val, label=val_target)

    print("computing best round...")
    evals_result = {}
    model = xgb.train(
        base_params, d_train, 
        num_boost_round=round_count,
        early_stopping_rounds=10,
        evals=[(d_val, "validation")],
        evals_result=evals_result,
        verbose_eval=False,
    )

    round_count = model.best_iteration + 1
    print("best round: {}".format(round_count))

    print("tuning max_depth and min_child_weight ...")
    max_depth_range = list(range(5,9))
    min_child_weight_range = list(range(1,4))
    tune_list = itertools.product(max_depth_range, min_child_weight_range)
    base_params = tune_xgb_param(
        model, d_train, d_val, base_params,
        tune_list, ['max_depth', 'min_child_weight'], round_count, loss_compare)

    print("tuning gamma ...")
    gamma_range = [(i/10.0,) for i in range(0,5)]
    tune_list = gamma_range
    base_params = tune_xgb_param(
        model, d_train, d_val, base_params,
        tune_list, ['gamma'], round_count, loss_compare)

    print("re-computing best round...")
    evals_result = {}
    model = xgb.train(
        base_params, d_train, 
        num_boost_round=round_count,
        early_stopping_rounds=10,
        evals=[(d_val, "validation")],
        evals_result=evals_result,
        verbose_eval=False,
    )

    round_count = model.best_iteration + 1
    print("best round: {}".format(round_count))


    print("tuning subsample and colsample_bytree ...")
    subsample_range = [i/10.0 for i in range(6,10)]
    colsample_bytree_range = [i/10.0 for i in range(0,10)]
    tune_list = itertools.product(subsample_range, colsample_bytree_range)
    base_params = tune_xgb_param(
        model, d_train, d_val, base_params,
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
        model, d_train, d_val, base_params,
        tune_list, ['subsample', 'colsample_bytree'],
        round_count, loss_compare)

    print("tuning alpha and lambda ...")
    alpha_range = [0, 0.01, 0.1, 0.5, 1]
    lambda_range = [1, 1.1, 1.5, 2, 5]

    tune_list = itertools.product(alpha_range, lambda_range)
    base_params = tune_xgb_param(
        model, d_train, d_val, base_params,
        tune_list, ['alpha', 'lambda'],
        round_count, loss_compare)

    print("tuning seed ...")
    seed_range = [(4,), (7,), (13,), (27,), (42,), (54,)]
    base_params = tune_xgb_param(
        model, d_train, d_val, base_params,
        seed_range, ['seed'], round_count, loss_compare)

    print(base_params)
    return base_params, round_count


def tune_xgb_model(
    x_train, y_train, x_val, y_val,
    params, max_round_count=5000, loss_compare=operator.lt, pass_count=2):
    '''Tunes a XGBoost model

    Examples:
        >>> params, round_count = tune_xgb_model(x, y, x_val, y_val, model_params)

    Args:
        x_train: Train set
        y_train: Train labels
        x_val: Validation set
        y_val: Validation labels
        params: A dictionary with the base xgboost parameters to use
        max_round_count: Maximum number of rounds during training
        pass_count: Number of tuning pass to do

    Returns:
        A tuple of tuned parameters and round count. 
    '''
    for tune_pass in range(pass_count):
        print('tuning pass {}...'.format(tune_pass))
        params, round_count = tune_xgb_pass(
            x_train, y_train,
            x_val, y_val,
            params, 
            round_count=max_round_count,
            loss_compare=loss_compare)

    return params, round_count
