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
        image.unit_features = image.unit_features.cpu().numpy()
        image.magnitude = image.magnitude.cpu().numpy()
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
            output_batch.unit_features = output_batch.unit_features.cpu().numpy()
            output_batch.magnitudes = output_batch.magnitudes.cpu().numpy()
            for idx, (unit_features, magnitude) in enumerate(zip(output_batch.unit_features, output_batch.magnitudes)):
                feat_idx = self.features_repo.add(unit_features)
                pg_img = pgImage(code=uuid.uuid4(), path=paths[idx], features_idx=feat_idx, magnitude=magnitude.item())
                self.current_db_session.add(pg_img)
                self.cnt_saved_images += 1
            self.features_repo.commit()
            self.current_db_session.commit()
            self.save_queue.task_done()
            self.stats.end("save_batch")

    def find_similars(self, image, topk=5):
        indices, values = self.features_repo.find_similars(image.unit_features, topk)
        indices = indices.cpu().numpy().tolist()
        values = values.cpu().numpy().tolist()
        DBSession = sessionmaker(bind=self.engine)
        db_session = DBSession()
        self.stats.start("query db")
        similars = db_session.query(pgImage).filter(pgImage.features_idx.in_(indices)).all()
        self.stats.end("query db")
        paths_dict = {}
        for s in similars:
            paths_dict[s.features_idx] = s.path
        result = []
        for i, v in zip(indices, values):
            result.append((paths_dict[i], v))
        logging.info(str(self.stats))
        return result

