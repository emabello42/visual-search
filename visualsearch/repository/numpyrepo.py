import numpy as np
import os
from visualsearch.configs import FeatureExtractorConfig as cfg

class NumpyRepo:

    def __init__(self, numpy_file):
        self.numpy_file = numpy_file
        if os.path.isfile(numpy_file):
            # it is more efficient to append features to a Python list (CPython implementation)
            # than working with numpy arrays, because the latter resize the
            # the array every time a new feature vector is appended and copy the whole array again.
            # More details: https://docs.python.org/2/faq/design.html#how-are-lists-implemented-in-cpython
            self.features_mat = np.load(self.numpy_file).tolist()
        else:
            self.features_mat = []

    def commit(self):
        np.save(self.numpy_file, np.array(self.features_mat))

    def add(self, features):
        self.features_mat.append(features)
        # self.features_mat = np.append(self.features_mat, np.array([features]), axis=0)
        feat_idx = len(self.features_mat) - 1
        return feat_idx
