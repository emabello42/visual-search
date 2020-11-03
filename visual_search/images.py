#!/usr/bin/env python3

import argparse
from application import db
import glob
from PIL import Image as PILImage
import torch
from database.models import Image, Category
import numpy as np
from model import ResnetExt

class VisualSearch:
    def __init__(self):
        self.model = ResnetExt()
        self.data_transform = transforms.Compose([transforms.Resize(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                 ])
        if torch.cuda.is_available():
            self.model.to('cuda')
        self.model.eval()

    def _compute_features(self, img):
        input_batch = self.data_transform(img).unsqueeze(0)
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            
        with torch.no_grad():
            logits, features = self.model(input_batch)[0]

        probs = torch.nn.functional.softmax(logits, dim=0)
        category_id = torch.argmax(probs).item()
        features = features.cpu().numpy()
        return features, category_id

    def save_image(self, path):
        img = PILImage.open(f)
        if img.size[0] > 299 and img.size[1] > 299:
            features, category_id = self._compute_features(img)
            category = db.session.query(Category).filter(Category.id == category_id).first()
            magnitude = np.linalg.norm(features)
            unit_features = features / magnitude
            new_img = Image(path=path, unit_features=unit_features.tolist(), magnitude=magnitude.item(), category=category)
            db.session.add(new_img)
            db.session.commit()

    def find_similarities(self, img):
        if img.size[0] > 299 and img.size[1] > 299:
            features, category_id = self._compute_features(img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save and find similar images')
    parser.add_argument("-s", "--save", required=False, type=str, help="Save images from a directory, computes their features and save them into the database")
    parser.add_argument("-f", "--find", required=False, type=str, help="Given an image, find similar ones in the database")
    args = vars(parser.parse_args())
    
    visualSearch = VisualSearch()

    if args['save']:
        image_files = glob.glob(f"{args['save']}/*.jpg")
        for f in image_files:
            visualSearch.save_image(f)
    elif args['find']:
        img = PILImage.open(args['find'])