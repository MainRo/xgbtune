==========
XGBTune
==========

XGBTune is a library for automated XGBoost model tunning.

Get Started
============

.. code:: python

    from xgbtune import tune_xgb_model

    params, round_count = tune_xgb_model(
        x_train, y_train,
        x_val, y_val,
        params)
