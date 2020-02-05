import os
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgbtune import tune_xgb_model


def test_titanic():
    dataset_path = str(pathlib.Path(__file__).parent.absolute())

    df = pd.read_csv(os.path.join(dataset_path, 'datasets', 'titanic.csv'))
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

    assert {} == actual_params
    assert 3 == actual_round_count
