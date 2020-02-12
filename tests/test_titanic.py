import os
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgbtune import tune_xgb_model


def test_titanic_val():
    dataset_path = str(pathlib.Path(__file__).parent.absolute())

    df = pd.read_csv(os.path.join(dataset_path, 'datasets', 'titanic.csv'))
    df['Sex'] = df['Sex'].map(lambda i: 1 if 'male' else 0)
    x = df[['Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    y = df['Survived']

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'error',
    }

    actual_params, actual_round_count = tune_xgb_model(
        params, x_train, y_train, x_val, y_val,
    )

    assert {
        'objective': 'binary:logistic',
        'eval_metric': 'error',
        'max_depth': 5,
        'min_child_weight': 1,
        'gamma': 0.4,
        'subsample': 0.9,
        'colsample_bytree': 1.0,
        'alpha': 0.0,
        'lambda': 1.0,
        'seed': 0} == actual_params
    assert 13 == actual_round_count


def test_titanic_cv():
    dataset_path = str(pathlib.Path(__file__).parent.absolute())

    df = pd.read_csv(os.path.join(dataset_path, 'datasets', 'titanic.csv'))
    df['Sex'] = df['Sex'].map(lambda i: 1 if 'male' else 0)
    x = df[['Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    y = df['Survived']

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'error',
    }

    actual_params, actual_round_count = tune_xgb_model(
        params, x, y, shuffle=False
    )

    assert {
        'objective': 'binary:logistic',
        'eval_metric': 'error',
        'max_depth': 8,
        'min_child_weight': 1,
        'gamma': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 1.0,
        'alpha': 0.01,
        'lambda': 1.1,
        'seed': 27} == actual_params
    assert 9 == actual_round_count
