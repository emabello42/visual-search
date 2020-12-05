import h5py
import os
import torch
import logging
from visualsearch.configs import FeatureExtractorConfig as Cfg
from visualsearch.utils import ProcessingStats


class HDF5Repo:

    def __init__(self, hdf5_file_name, mode="a"):
        self.hdf5_file = h5py.File(hdf5_file_name, mode)
        self.use_gpu = torch.cuda.is_available()
        self.stats = ProcessingStats()

    def close(self):
        self.hdf5_file.close()

    def reserve_space(self, image_data_size):
        if "features" not in self.hdf5_file:
            self.features_set = self.hdf5_file.create_dataset(
                                                 "features", (image_data_size, Cfg.feature_size),
                                                 chunks=(Cfg.batch_size, Cfg.feature_size),
                                                 maxshape=(None, Cfg.feature_size))
            self.features_set.attrs['next_idx'] = 0  # indicates the index where is going to be placed the next vector
        else:
            self.features_set = self.hdf5_file["features"]
            self.features_set.resize(self.features_set.shape[0] + image_data_size, axis=0)

    def add(self, features):
        feat_idx = self.features_set.attrs['next_idx']
        self.features_set[feat_idx, :] = features
        self.features_set.attrs['next_idx'] += 1
        return feat_idx.item()

    def find_similars(self, features_v, topk):
        self.stats.start("numpyrepo - find similars")
        self.stats.start("numpyrepo - conversion to tensor")
        features_mat = torch.from_numpy(self.hdf5_file["features"][:, :])
        self.stats.end("numpyrepo - conversion to tensor")
        if self.use_gpu:
            features_mat = features_mat.cuda()
            features_v = features_v.cuda()
        self.stats.start("numpyrepo - matrix multiplication")
        logging.info(features_mat.dtype)
        logging.info(features_v.dtype)
        similarities_v = torch.matmul(features_mat, features_v.T)
        self.stats.end("numpyrepo - matrix multiplication")
        self.stats.start("numpyrepo - top k")
        top_similarities = torch.topk(similarities_v, topk)
        self.stats.end("numpyrepo - top k")
        self.stats.end("numpyrepo - find similars")
        logging.info(str(self.stats))
        return top_similarities.indices.cpu(), top_similarities.values.cpu()

