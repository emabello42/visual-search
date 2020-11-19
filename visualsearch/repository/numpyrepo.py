import numpy as np
import os
from visualsearch.configs import FeatureExtractorConfig as cfg
import logging

class NumpyRepo:

    def __init__(self, numpy_file):
        self.numpy_file = numpy_file
        if os.path.isfile(numpy_file):
            self.features_mat = np.load(self.numpy_file)
        else:
            self.features_mat = np.empty((0, cfg.feature_size), float)

    def __del__(self):
        logging.debug("Saving numpy file {}".format(self.numpy_file))
        np.save(self.numpy_file, self.features_mat)

    def save(self, features):
        self.features_mat = np.append(self.features_mat, np.array([features]), axis=0)
        feat_idx = len(self.features_mat) - 1
        return feat_idx
