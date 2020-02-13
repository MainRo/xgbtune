.. parameters:

Parameters
===========

Configuring cross-validation
-----------------------------

The following parameters are directly forwarded to the xgboost.cv function call:

* nfold: number of folds
* stratified: Perform stratified sampling
* folds: Sklearn KFolds or StratifiedKFolds object
* shuffle: shuffle data

These parameters must be provided as keyword parameters. Here is an example of
tuning with 5 folds and without shuffling:

.. code:: python

    from xgbtune import tune_xgb_model

    params, round_count = tune_xgb_model(
        params,
        x_train, y_train,
        folds=5, shuffle=False)


Using a validation set
-----------------------

It is possible to use a validation set instead of cross-validation:

.. code:: python

    from xgbtune import tune_xgb_model

    params, round_count = tune_xgb_model(
        params,
        x_train, y_train,
        x_val, y_val)


Providing your own hyper-parameter search space
------------------------------------------------

The tune_params is a dict that can be used to overload the default search space
for each parameter. A search space is provided as a list, with the dict key name
corresponding to the xgboost parameter: max_depth, min_child_weight, gamma,
subsample, colsample_bytree, alpha, lambda, seed.

Here is an example of tuning with custom alpha search space:

.. code:: python

    from xgbtune import tune_xgb_model

    params, round_count = tune_xgb_model(
        params,
        x_train, y_train,
        {"alpha": [(i/10.0,) for i in range(0,11)]})
