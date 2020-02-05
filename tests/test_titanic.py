import pandas as pd
from learn.model_selection import train_test_split
from xgbtune import tune_xgb_model


def test_titanic():
    df = pd.read_csv('./datasets/titanic.csv')
    x = df[['Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    y = df['Survived']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    params = {

    }

    actual_params, actual_round_count = tune_xgb_model(x_train, y_train, x_test, y_test)
    assert {} == actual_params
    assert 3 == actual_round_count