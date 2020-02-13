Get Started
============

XGBTune uses a default configuration that should work in most cases. This allows
to tune a model by using cross compilation‡ very easily:

.. code:: python

    from xgbtune import tune_xgb_model

    params, round_count = tune_xgb_model(params, x_train, y_train)

params contains your xgboost configuration and initial parameters. x_train and
y_train are your datasets with labels. The result is a tuple with the tuned
parameters and the optimal number of booster rounds.