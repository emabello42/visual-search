from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

class DBConfig:
    driver = "postgres"
    db = "visualsearch"
    user = "visualsearch"
    password = "visualsearch"
    host = "localhost"
    port = 5432

    @staticmethod
    def get_uri():
        return f"{DBConfig.driver}://{DBConfig.user}:{DBConfig.password}@{DBConfig.host}:{DBConfig.port}/{DBConfig.db}"

# initialization
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DBConfig.get_uri()
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# extensions
db = SQLAlchemy(app)