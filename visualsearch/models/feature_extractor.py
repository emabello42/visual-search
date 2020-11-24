from visualsearch.models.resnetext import ResnetExt
from visualsearch.utils import ProcessingStats
from torchvision import transforms
import torch
from PIL import Image as PILImage
from visualsearch.configs import FeatureExtractorConfig as Cfg
from collections import namedtuple
from torch.utils.data import Dataset
import os
import uuid
from visualsearch.domain.image import Image
import logging


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
            self.model = self.model.cuda()
        self.model.eval()
        self.stats = ProcessingStats()

    def process_image(self, path):
        img = PILImage.open(path)
        data = self.data_transform(img).unsqueeze(0)
        output_batch = self.__compute_features(data)

        new_image = Image(
            code=uuid.uuid4(),
            path=path,
            unit_features=output_batch.unit_features[0],
            magnitude=output_batch.magnitudes[0].item()
        )

        return new_image

    def process_batch(self, path):
        image_data = CustomImageDataSet(path, transform=self.data_transform)
        image_loader = torch.utils.data.DataLoader(image_data, Cfg.batch_size, shuffle=False,
                                                   num_workers=Cfg.num_workers)

        for batch_idx, (data, paths) in enumerate(image_loader):
            output_batch = self.__compute_features(data)
            yield output_batch, paths
        logging.debug(str(self.stats))

    def __compute_features(self, input_batch):
        self.stats.start("compute_features")
        output_batch = namedtuple("BatchFeatures", "unit_features magnitudes")
        if self.use_gpu:
            input_batch = input_batch.cuda()
        with torch.no_grad():
            unit_features, magnitudes = self.model(input_batch)
        output_batch.unit_features = unit_features
        output_batch.magnitudes = magnitudes
        self.stats.end("compute_features")
        return output_batch
