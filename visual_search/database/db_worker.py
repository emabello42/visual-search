import queue
import time
from application import db
from database.models import Image, Category
import logging
import utils

class DatabaseWorker:
    def __init__(self):
        self.qImgData = queue.Queue(10000)

    def save_image_batch(self):
        session = db.create_scoped_session() # create a new DB session for this thread
        categories = session.query(Category).all()
        categories_map = {}
        for c in categories:
            categories_map[c.id] = c
        while True:
            (paths, unit_features, magnitudes, category_ids, scores) = self.qImgData.get()
            start = time.time()
            for path, unit_feat, mag, cat_id, score in zip(paths, unit_features, magnitudes, category_ids, scores):
                new_img = Image(path=path,
                                unit_features_data=utils.adapt_array(unit_feat),
                                magnitude=mag.item(),
                                category=categories_map[cat_id],
                                score=score.item())
                session.add(new_img)
            session.commit()
            end = time.time()
            logging.debug("Queue size: " + str(self.qImgData.qsize()))
            logging.debug("Batch saved in database: " + str(end-start) + "s")
            self.qImgData.task_done()