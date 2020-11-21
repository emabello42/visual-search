from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from visualsearch.repository.postgres_objects import Base, Image as pgImage
from visualsearch.repository.numpyrepo import NumpyRepo
import queue
import threading
import uuid
from visualsearch.utils import ProcessingStats
import logging

class PostgresRepo:
    def __init__(self, connection_data, features_file):
        connection_string = "postgresql+psycopg2://{}:{}@{}/{}".format(
            connection_data['user'],
            connection_data['password'],
            connection_data['host'],
            connection_data['dbname']
        )
        self.stats = ProcessingStats()
        self.engine = create_engine(connection_string)
        Base.metadata.bind = self.engine
        self.features_repo = NumpyRepo(features_file)
        self.save_queue = queue.Queue(10000)
        self.current_db_session = None
        self.cnt_saved_images = 0

    def save_image(self, image):
        DBSession = sessionmaker(bind=self.engine)
        session = DBSession()
        feat_idx = self.features_repo.add(image.unit_features)
        pg_img = pgImage(code=image.code, path=image.path, features_idx=feat_idx, magnitude=image.magnitude.item())
        session.add(pg_img)
        self.features_repo.commit()
        self.session.commit()
        return 1

    def start_save_batch_process(self):
        threading.Thread(target=self.__save_batch, daemon=True).start()
        DBSession = sessionmaker(bind=self.engine)
        self.current_db_session = DBSession()

    def close_save_batch_process(self):
        self.save_queue.join()
        logging.debug(str(self.stats))
        return self.cnt_saved_images

    def __save_batch(self):
        while True:
            output_batch, paths = self.save_queue.get()
            self.stats.start("save_batch")
            for idx, (unit_features, magnitude) in enumerate(zip(output_batch.unit_features, output_batch.magnitudes)):
                feat_idx = self.features_repo.add(unit_features)
                pg_img = pgImage(code=uuid.uuid4(), path=paths[idx], features_idx=feat_idx, magnitude=magnitude.item())
                self.current_db_session.add(pg_img)
                self.cnt_saved_images += 1
            self.features_repo.commit()
            self.current_db_session.commit()
            self.save_queue.task_done()
            self.stats.end("save_batch")

    def find_similars(self, image):
        pass
