import pytest
from visualsearch.app import create_app
from visualsearch.flask_settings import TestConfig

@pytest.yield_fixture(scope='function')
def app():
    return create_app(TestConfig)
