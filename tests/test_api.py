import pytest
from api import app, ROOT_DIR
import pandas as pd

@pytest.fixture
def df_test():
    test_file = f'{ROOT_DIR}/exercise_26_test.csv'
    df_test = pd.read_csv(test_file)
    yield df_test


@pytest.fixture
def client():
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client

def test_home(client):
    """Test that the api is reachable"""
    rv = client.get('/')

    assert rv.status_code == 200

def assert_row(json):
    """
    Asserts that a row from the response payload is of the correct type and
    contains the appropriate data.
    :param json: Row of output
    """
    # ensure correct variables were returned
    assert 'business_outcome' in json
    assert 'phat' in json
    assert 'params' in json

    # ensure variables are of the correct type
    assert type(json['business_outcome']) == int
    assert type(json['phat']) == float
    assert type(json['params']) == dict


def test_one_row(client, df_test):
    """
    Tests that the /predict route can successfully accept one row
    of test data and return three variables : business_outcome, phat, and params.
    The output must be in JSON format.
    :param client: connection to api
    """
    # get first row
    data = df_test.iloc[0].to_dict()

    rv = client.post('/predict', json=data)

    assert rv.status_code == 200

    assert rv.is_json == True

    jsonRes = rv.get_json()

    assert_row(jsonRes)


def test_five_rows(client, df_test):
    """
    Tests that the /predict route can successfully accept five rows
    of test data and return an array with the business_outcome, phat,
    and params variables in JSON.
    :param client: connection to api
    """
    nRows = 5

    # get first five row
    data = df_test.head().to_dict('records')

    rv = client.post('/predict', json=data)

    assert rv.status_code == 200

    assert rv.is_json == True

    jsonRes = rv.get_json()

    assert len(jsonRes) == nRows

    for obj in jsonRes:
        assert_row(obj)

def test_all_rows(client, df_test):
    """
    Tests that the /predict route can successfully accept all rows
    of the test data and return an array with the business_outcome, phat,
    and params variables in JSON.
    :param client: connection to api
    """

    nRows = df_test.shape[0]

    # get first five row
    data = df_test.to_dict('records')

    rv = client.post('/predict', json=data)

    assert rv.status_code == 200

    assert rv.is_json == True

    jsonRes = rv.get_json()

    assert len(jsonRes) == nRows

    for obj in jsonRes:
        assert_row(obj)

def test_no_data(client):
    """
    Asserts API can handle invalid data
    :param client: connection to api
    :return:
    """

    rv = client.post('/predict', json='')

    assert rv.status_code == 400

    print("\n", rv.get_json())