#!/usr/bin/env python3

import argparse
from application import db
import glob
from PIL import Image as PILImage
import torch
from database.models import Image, Category
import numpy as np
from model import ResnetExt
from torchvision import datasets
from torchvision import transforms
import time
import logging
import threading
from database.db_worker import DatabaseWorker
from utils import convert_array, ProcessingStats
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class VisualSearch:
    def __init__(self):
        self.model = ResnetExt()
        self.data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                                                       0.229, 0.224, 0.225]),
                                                  ])
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.to('cuda')
        self.model.eval()
        self.pstats = ProcessingStats()

    def _compute_features(self, input_batch):
        self.pstats.start("compute_features")
        output_batch = namedtuple(
            "BatchFeatures", "unit_features magnitudes scores category_ids")
        if self.use_gpu:
            input_batch = input_batch.to('cuda')
        with torch.no_grad():
            logits, unit_features, magnitudes = self.model(input_batch)

        probabilities = torch.nn.functional.softmax(logits, dim=1)
        category_ids = torch.argmax(probabilities, dim=1)

        probabilities = probabilities.cpu().numpy()
        category_ids = category_ids.cpu().numpy()
        scores = []
        for i, p in enumerate(probabilities):
            scores.append(p[category_ids[i]])

        output_batch.unit_features = unit_features.cpu().numpy()
        output_batch.magnitudes = magnitudes.cpu().numpy()
        output_batch.scores = np.array(scores)
        output_batch.category_ids = category_ids
        self.pstats.end("compute_features")
        return output_batch

    # def save_image(self, path):
    #     img = PILImage.open(f)
    #     if img.size[0] > 299 and img.size[1] > 299:
    #         features, category_id = self._compute_features(img)
    #         category = db.session.query(Category).filter(
    #             Category.id == category_id).first()
    #         magnitude = np.linalg.norm(features)
    #         unit_features = features / magnitude
    #         new_img = Image(path=path, unit_features=unit_features.tolist(
    #         ), magnitude=magnitude.item(), category=category)
    #         db.session.add(new_img)
    #         db.session.commit()

    def save_image_batch(self, path, batch_size=256):
        db_worker = DatabaseWorker()

        image_data = datasets.ImageFolder(path, transform=self.data_transform)
        image_loader = torch.utils.data.DataLoader(
            image_data, batch_size, shuffle=False, num_workers=8)

        # turn-on the worker thread
        threading.Thread(target=db_worker.save_image_batch,
                         daemon=True).start()

        img_idx = 0
        for batch_idx, (data, _) in enumerate(image_loader):
            start = time.time()
            batch_features = self._compute_features(data)
            paths = [image_data.imgs[i][0] for i in range(
                img_idx, min(batch_size+img_idx, len(image_data.imgs)))]
            img_idx += batch_size

            batch_features.paths = paths
            db_worker.qImgData.put(batch_features)
            end = time.time()
            logging.debug("Batch " + str(batch_idx) +
                          " processed in " + str(end-start) + "s")

        # block until all tasks are done
        db_worker.qImgData.join()

    def find_similarities(self, img):
        self.pstats.start("find_similarities")
        data = self.data_transform(img).unsqueeze(0)
        img_features = self._compute_features(data)
        img_category_label = img_features.category_ids[0].item()
        logging.debug("Category found: " + str(img_category_label))
        img_ufeats = img_features.unit_features[0]
        self.pstats.start("query_images_with_id")
        images = db.session.query(Image).join(Category, Category.id == Image.category_id).filter(
            Category.label == img_category_label).all()
        self.pstats.end("query_images_with_id")
        if len(images) == 0:
            self.pstats.start("query_all_images")
            images = db.session.query(Image).all()
            self.pstats.end("query_all_images")
        images_dict = {img.id: img for img in images}
        similarities = []
        for img in images:
            self.pstats.start("convert array of features")
            ufeats = convert_array(img.unit_features)
            self.pstats.end("convert array of features")
            self.pstats.start("dot product of unit features")
            cosine = img_ufeats.dot(ufeats)
            self.pstats.end("dot product of unit features")
            similarities.append((img, cosine))

        similarities = [e for e in sorted(
            similarities, key=lambda item: item[1])]
        self.pstats.end("find_similarities")
        width = 5
        height = 5
        rows = 2
        cols = 2
        axes = []
        fig = plt.figure()
        idx = 1
        for s in similarities[-4:]:
            similar_img = mpimg.imread(s[0].path)
            axes.append(fig.add_subplot(rows, cols, idx))
            subplot_title = ("cosine: "+str(s[1]))
            axes[-1].set_title(subplot_title)
            plt.imshow(similar_img)
            idx += 1

        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Save and find similar images')
    parser.add_argument("-s", "--save", required=False, type=str,
                        help="Save images from a directory, computes their features and save them into the database")
    parser.add_argument("-f", "--find", required=False, type=str,
                        help="Given an image, find similar ones in the database")
    args = vars(parser.parse_args())

    visualSearch = VisualSearch()

    if args['save']:
        visualSearch.save_image_batch(args['save'])
        print('All work completed')
    elif args['find']:
        img = PILImage.open(args['find'])
        visualSearch.find_similarities(img)
        logging.info(visualSearch.pstats)
