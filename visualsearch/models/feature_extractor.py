from visualsearch.models.resnetext import ResnetExt
from visualsearch.utils import ProcessingStats
from torchvision import datasets
from torchvision import transforms
import torch
from PIL import Image as PILImage
from collections import namedtuple
import numpy as np

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
        output_batch = self._compute_features(data)

        features = {
                'unit_features': output_batch.unit_features[0],
                'magnitude': output_batch.magnitudes[0].item(),
                'label': output_batch.category_ids[0],
                'score': output_batch.scores[0]
                }

        return features

    def process_batch(self, path):
        pass

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
