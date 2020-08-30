import pytest
from api import app
from flask import Response

@pytest.fixture
def client():
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client

def test_empty(client):
    """Start with a blank database."""

    rv = client.get('/')

    assert rv.status_code == 200