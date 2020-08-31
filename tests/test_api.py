import pytest
from api import app, ROOT_DIR
import pandas as pd

def load_test_data():
    test_file = f'{ROOT_DIR}/exercise_26_test.csv'
    df_test = pd.read_csv(test_file)
    return df_test


@pytest.fixture
def client():
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client

def test_empty(client):
    """Test that the api is reachable"""
    rv = client.get('/')

    assert rv.status_code == 200

def test_one_row(client):
    """
    Tests that the /predict route can successfully accept one row
    of parameters and return three variables : business_outcome, phat, and params.
    The output must be in JSON format.
    :param client: connection to api
    """
    df_test = load_test_data()

    # get first row
    data = df_test.iloc[0].to_dict()

    rv = client.post('/predict', json=data)

    assert rv.status_code == 200

    assert rv.is_json == True

    jsonRes = rv.get_json()

    assert 'business_outcome' in jsonRes
    assert 'phat' in jsonRes
    assert 'params' in jsonRes

    # print(json)
