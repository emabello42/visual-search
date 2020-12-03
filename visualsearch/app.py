from flask import Flask
import logging
from visualsearch.rest import find_similarities
from visualsearch.flask_settings import ProdConfig

logging.basicConfig(level=logging.INFO,
                    format='(%(threadName)-9s) %(message)s', )


def create_app(config_object=ProdConfig):
    app = Flask(__name__)
    app.config.from_object(config_object)
    app.register_blueprint(find_similarities.blueprint)
    return app
