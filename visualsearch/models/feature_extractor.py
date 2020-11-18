from visualsearch.models.resnetext import ResnetExt
from visualsearch.utils import ProcessingStats
from torchvision import datasets
from torchvision import transforms
import torch
from PIL import Image as PILImage
import numpy as np
from dataclasses import dataclass
from visualsearch.configs import FeatureExtractorConfig as cfg
from typing import List
from collections import namedtuple
from torch.utils.data import Dataset
import os

@dataclass
class ImageFeatures:
    unit_features: np.ndarray = None
    magnitude: float = None
    score: float = None
    label: int = None
    path: str = None


class CustomImageDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.imgs[idx])
        image = PILImage.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, img_loc


class FeatureExtractor():

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

    def process_image(self, path):
        img = PILImage.open(path)
        data = self.data_transform(img).unsqueeze(0)
        output_batch = self.__compute_features(data)

        img_features = ImageFeatures(
                unit_features = output_batch.unit_features[0],
                magnitude = output_batch.magnitudes[0].item(),
                label = output_batch.labels[0],
                score = output_batch.scores[0],
                path = path
                )

        return img_features

    def process_batch(self, path):
        image_data = CustomImageDataSet(path, transform=self.data_transform)
        image_loader = torch.utils.data.DataLoader(image_data, cfg.batch_size, shuffle=False,
                                                   num_workers=cfg.num_workers)

        for batch_idx, (data, paths) in enumerate(image_loader):
            output_batch = self.__compute_features(data)
            for img_idx, (unit_feat, mag, label, score) in enumerate(zip(output_batch.unit_features,
                                                            output_batch.magnitudes,
                                                            output_batch.labels,
                                                            output_batch.scores)):
                img_features = ImageFeatures(unit_features = unit_feat,
                                             magnitude = mag,
                                             label = label,
                                             score = score,
                                             path = paths[img_idx])
                yield img_features

    def __compute_features(self, input_batch):
        self.pstats.start("compute_features")
        output_batch = namedtuple("BatchFeatures",
                                  "unit_features magnitudes scores labels")
        if self.use_gpu:
            input_batch = input_batch.to('cuda')
        with torch.no_grad():
            logits, unit_features, magnitudes = self.model(input_batch)

        probabilities = torch.nn.functional.softmax(logits, dim=1)
        labels = torch.argmax(probabilities, dim=1)

        probabilities = probabilities.cpu().numpy()
        labels = labels.cpu().numpy()
        scores = []
        for i, p in enumerate(probabilities):
            scores.append(p[labels[i]])

        output_batch.unit_features = unit_features.cpu().numpy()
        output_batch.magnitudes = magnitudes.cpu().numpy()
        output_batch.scores = np.array(scores)
        output_batch.labels = labels
        self.pstats.end("compute_features")
        return output_batch

