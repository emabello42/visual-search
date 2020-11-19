import sqlalchemy
import sqlalchemy_utils

from visualsearch.repository.postgres_objects import Base

setup = {
    'dbname': 'visualsearch',
    'user': 'visualsearch',
    'password': 'visualsearch',
    'host': 'localhost'
}

conn_str = "postgresql+psycopg2://{}:{}@{}/{}".format(
    setup['user'],
    setup['password'],
    setup['host'],
    setup['dbname']
)

engine = sqlalchemy.create_engine(conn_str)
if not sqlalchemy_utils.database_exists(engine.url):
    sqlalchemy_utils.create_database(engine.url)

conn = engine.connect()

Base.metadata.create_all(engine)
