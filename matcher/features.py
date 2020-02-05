import pickle
import torch
from PIL import Image
import time
from sklearn.neighbors import KDTree
import numpy as np
import torchvision.transforms.functional as TF
from matcher.models import set_as_feature_extractor


class FeatureMatcher:

    def __init__(self, model_path, features_path, index_path):
        with open(index_path, "rb") as pic:
            self.index = pickle.load(pic)

        print("Loading model")
        self.model = torch.load(model_path)
        # self.model.set_feature_extractor()
        set_as_feature_extractor(self.model)

        t0 = time.time()
        features_matrix = np.squeeze(np.load(features_path))
        self.tree = KDTree(features_matrix)
        print("Loaded in {}s".format(time.time() - t0))

    def get_k_most_similar(self, input_path, image_size, k=1):
        image = Image.open(input_path).convert('RGB').resize(image_size)
        image.show()

        x = TF.to_tensor(image)
        feature = self.model(x.unsqueeze(0))
        print('Loading features')

        t0 = time.time()
        print("Looking for ...")
        similar = self.tree.query(feature, k=k, return_distance=False)

        print("Found in {}s".format(time.time() - t0))

        result = []
        for i in similar[0]:
            result.append(self.index[i][:-4])

        return result
