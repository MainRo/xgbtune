==========
XGBTune
==========


.. image:: https://badge.fury.io/py/xgbtune.svg
    :target: https://badge.fury.io/py/xgbtune

.. image:: https://github.com/mainro/xgbtune/workflows/Python%20package/badge.svg
    :target: https://github.com/mainro/xgbtune/actions?query=workflow%3A%22Python+package%22
    :alt: Github WorkFlows

.. image:: https://readthedocs.org/projects/xgbtune/badge/?version=latest
    :target: https://xgbtune.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


XGBTune is a library for automated XGBoost model tunning. Tunning an XGBoost
model is as simple as a single function call.

Get Started
============

.. code:: python

    from xgbtune import tune_xgb_model

    params, round_count = tune_xgb_model(params, x_train, y_train)
