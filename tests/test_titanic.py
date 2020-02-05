import os
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgbtune import tune_xgb_model


def test_titanic():
    dataset_path = str(pathlib.Path(__file__).parent.absolute())

    df = pd.read_csv(os.path.join(dataset_path, 'datasets', 'titanic.csv'))
    df['Sex'] = df['Sex'].map(lambda i: 1 if 'male' else 0)
    x = df[['Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    y = df['Survived']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'error',
    }

    actual_params, actual_round_count = tune_xgb_model(
        x_train, y_train, x_test, y_test,
        params
    )

    assert {
        'objective': 'binary:logistic',
        'eval_metric': 'error',
        'max_depth': 8,
        'min_child_weight': 1,
        'gamma': 0.4,
        'subsample': 0.6,
        'colsample_bytree': 0.8,
        'alpha': 0.1,
        'lambda': 1.5,
        'seed': 54} == actual_params
    assert 11 == actual_round_count
