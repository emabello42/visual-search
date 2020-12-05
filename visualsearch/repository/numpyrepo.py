import numpy as np
import os
import torch
import logging
from visualsearch.utils import ProcessingStats

class NumpyRepo:

    def __init__(self, numpy_file):
        self.use_gpu = torch.cuda.is_available()
        self.numpy_file = numpy_file
        self.stats = ProcessingStats()
        if os.path.isfile(numpy_file):
            # it is more efficient to append features to a Python list (CPython implementation)
            # than working with numpy arrays, because the latter resize the
            # the array every time a new feature vector is appended and copy the whole array again.
            # More details: https://docs.python.org/2/faq/design.html#how-are-lists-implemented-in-cpython
            self.np_features_mat = np.load(self.numpy_file)
            self.features_mat = self.np_features_mat.tolist()
        else:
            self.features_mat = []

    def commit(self):
        np.save(self.numpy_file, np.array(self.features_mat, dtype=np.float32))

    def add(self, features):
        self.features_mat.append(features)
        # self.features_mat = np.append(self.features_mat, np.array([features]), axis=0)
        feat_idx = len(self.features_mat) - 1
        return feat_idx

    def find_similars(self, features_v, topk):
        self.stats.start("numpyrepo - find similars")
        self.stats.start("numpyrepo - conversion to tensor")
        features_mat = torch.from_numpy(self.np_features_mat)
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

