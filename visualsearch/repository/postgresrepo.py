from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from visualsearch.repository.postgres_objects import Base, Image as pgImage
import visualsearch.domain as domain
from visualsearch.repository.numpyrepo import NumpyRepo

class PostgresRepo:

    def __init__(self, connection_data, features_file):
        connection_string = "postgresql+psycopg2://{}:{}@{}/{}".format(
                connection_data['user'],
                connection_data['password'],
                connection_data['host'],
                connection_data['dbname']
        )
        self.engine = create_engine(connection_string)
        Base.metadata.bind = self.engine
        self.features_repo = NumpyRepo(features_file)

    def save_image(self, image):
        DBSession = sessionmaker(bind=self.engine)
        session = DBSession()
        feat_idx = self.features_repo.save(image.unit_features)
        pg_img = pgImage(code= image.code, path= image.path, features_idx=feat_idx, magnitude=image.magnitude.item())
        session.add(pg_img)
        session.commit()
        return 1

    def find_similars(self, image):
        pass

