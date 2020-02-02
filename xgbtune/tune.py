import itertools
import xgboost as xgb

def tune_xgb_param(model, train_set, val_set, parameters, tune_list, tune_params, round_count):
    best_loss = None
    best_config = None
    for tune_config in tune_list:
        for index, param in enumerate(tune_params):
            parameters[param] = tune_config[index]
        evals_result = {}
        model = xgb.train(
            parameters, train_set,
            num_boost_round=round_count,
            early_stopping_rounds=10,
            evals=[(val_set, "validation")],
            evals_result=evals_result,
            verbose_eval=False,
        )

        loss = evals_result['validation']['rmse'][-1]
        if best_loss == None or loss < best_loss:
            print("new best loss {} with {}".format(loss, parameters))
            best_loss = loss
            best_config = tune_config
        else:
            print("not best loss {}".format(loss))
    
        for index, param in enumerate(tune_params):
            parameters[param] = best_config[index]

    return parameters


def tune_xgb_pass(train, target, val, val_target, base_params, round_count):
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

    round_count = model.best_iteration
    print("best round: {}".format(round_count))

    print("tuning max_depth and min_child_weight ...")
    max_depth_range = list(range(5,9))
    min_child_weight_range = list(range(1,4))
    tune_list = itertools.product(max_depth_range, min_child_weight_range)
    base_params = tune_xgb_param(
        model, d_train, d_val, base_params,
        tune_list, ['max_depth', 'min_child_weight'], round_count)

    print(base_params)

    print("tuning gamma ...")
    gamma_range = [i/10.0 for i in range(0,5)]
    tune_list = gamma_range
    base_params = tune_xgb_param(
        model, d_train, d_val, base_params,
        tune_list, ['gamma'], round_count)

    print(base_params)

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

    round_count = model.best_iteration
    print("best round: {}".format(round_count))


    print("tuning subsample and colsample_bytree ...")
    subsample_range = [i/10.0 for i in range(6,10)]
    colsample_bytree_range = [i/10.0 for i in range(6,10)]
    tune_list = itertools.product(subsample_range, colsample_bytree_range)
    base_params = tune_xgb_param(
        model, d_train, d_val, base_params,
        tune_list, ['subsample', 'colsample_bytree'], round_count)

    print(base_params)

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
    best_loss = 5000
    best_config = None
    for tune_config in tune_list:
        params = base_params.copy()
        params['subsample'] = tune_config[0]
        params['colsample_bytree'] = tune_config[1]
        evals_result = {}
        model = xgb.train(
            params, d_train, 
            num_boost_round=round_count,
            early_stopping_rounds=10,
            evals=[(d_val, "validation")],
            evals_result=evals_result,
            verbose_eval=False,
        )

        #print(evals_result)
        loss = evals_result['validation']['rmse'][-1]
        if loss < best_loss:
            print("new best loss {} with subsample={}, colsample_bytree={}".format(
                loss, params['subsample'], params['colsample_bytree']
            ))
            best_loss = loss
            best_config = tune_config
    
    base_params['subsample'] = best_config[0]
    base_params['colsample_bytree'] = best_config[1]

    print("tuning alpha and lambda ...")
    alpha_range = [0, 0.01, 0.1, 0.5, 1]
    lambda_range = [1, 1.1, 1.5, 2, 5]

    tune_list = itertools.product(alpha_range, lambda_range)
    best_loss = 5000
    best_config = None
    for tune_config in tune_list:
        params = base_params.copy()
        params['alpha'] = tune_config[0]
        params['lambda'] = tune_config[1]
        evals_result = {}
        model = xgb.train(
            params, d_train, 
            num_boost_round=round_count,
            early_stopping_rounds=10,
            evals=[(d_val, "validation")],
            evals_result=evals_result,
            verbose_eval=False,
        )

        loss = evals_result['validation']['rmse'][-1]
        if loss < best_loss:
            print("new best loss {} with alpha={}, lambda={}".format(
                loss, params['alpha'], params['lambda']
            ))
            best_loss = loss
            best_config = tune_config
    
    base_params['alpha'] = best_config[0]
    base_params['lambda'] = best_config[1]


    print("tuning seed ...")    
    seed_range = [(4,), (7,), (13,), (27,), (42,), (54,)]
    base_params = tune_xgb_param(
        model, d_train, d_val, base_params,
        seed_range, ['seed'], round_count)

    print(base_params)
    return base_params, round_count


def tune_xgb_model(x_train, y_train, x_val, y_val, params, round_count=5000, pass_count=2):
    '''Tunes a XGBoost model

    Examples:
        >>> params, round_count = tune_xgb_model(x, y, x_val, y_val, model_params)

    Args:
        x_train: Train set
        y_train: Train labels
        x_val: Validation set
        y_val: Validation labels
        params: A dictionary with the base xgboost parameters to use
        round_count: Maximum number of rounds during training
        pass_count: Number of tuning pass to do

    Returns:
        A tuple of tuned parameters and round count. 
    '''
    for tune_pass in range(pass_count):
        print('tuning pass {}...'.format(tune_pass))
        params, round_count = tune_xgb_pass(
            x_train, y_train,
            x_val, y_val,
            params, round_count)

    return params, round_count
