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

    def _compute_features(self, input_batch):
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

        unit_features = unit_features.cpu().numpy()
        magnitudes = magnitudes.cpu().numpy()
        scores = np.array(scores)
        return unit_features, magnitudes, category_ids, scores

    def save_image(self, path):
        img = PILImage.open(f)
        if img.size[0] > 299 and img.size[1] > 299:
            features, category_id = self._compute_features(img)
            category = db.session.query(Category).filter(
                Category.id == category_id).first()
            magnitude = np.linalg.norm(features)
            unit_features = features / magnitude
            new_img = Image(path=path, unit_features=unit_features.tolist(
            ), magnitude=magnitude.item(), category=category)
            db.session.add(new_img)
            db.session.commit()

    def save_image_batch(self, path, batch_size=256):
        categories = db.session.query(Category).all()
        categories_map = {}
        for c in categories:
            categories_map[c.id] = c

        image_data = datasets.ImageFolder(path, transform=self.data_transform)
        image_loader = torch.utils.data.DataLoader(
            image_data, batch_size, shuffle=False, num_workers=0)

        img_idx = 0
        for batch_idx, (data, _) in enumerate(image_loader):
            start = time.time()
            unit_features, magnitudes, category_ids, scores = self._compute_features(
                data)
            end1 = time.time()
            img_bulk = []
            for unit_feat, mag, cat_id, score in zip(unit_features, magnitudes, category_ids, scores):
                new_img = Image(path=image_data.imgs[img_idx][0],
                                unit_features=unit_feat.tolist(),
                                magnitude=mag.item(),
                                category=categories_map[cat_id],
                                score=score.item())
                img_bulk.append(new_img)
                img_idx += 1
            end2 = time.time()
            db.session.bulk_save_objects(img_bulk)
            end3 = time.time()
            db.session.commit()
            end = time.time()
            print("Batch ", batch_idx, " processed in ", end-start, "s")
            print("compute features: ", end1 - start)
            print("create images objects: ", end2 - end1)
            print("save to database: ", end-end2)
            print("commit database: ", end-end3)

    def find_similarities(self, img):
        if img.size[0] > 299 and img.size[1] > 299:
            features, category_id = self._compute_features(img)


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
        # image_files = glob.glob(f"{args['save']}/*.jpg")
        # for f in image_files:
        #     visualSearch.save_image(f)
    elif args['find']:
        img = PILImage.open(args['find'])
