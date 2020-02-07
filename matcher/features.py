import pickle
import torch
from PIL import Image
import time
from sklearn.neighbors import KDTree
import numpy as np
import torchvision.transforms.functional as TF
from matcher.models import set_as_feature_extractor
import cv2
from matplotlib import pyplot as plt


class FeatureMatcher:
    def __init__(self, model_path, features_path, index_path, segmentation_model_path):
        with open(index_path, "rb") as pic:
            self.index = pickle.load(pic)

        print("Loading model")
        self.model_path = model_path
        self.segmentation_model_path = segmentation_model_path

        # self.model.set_feature_extractor()

        t0 = time.time()
        features_matrix = np.squeeze(np.load(features_path))
        self.tree = KDTree(features_matrix)
        print("Loaded in {}s".format(time.time() - t0))

    def get_k_most_similar(self, input_path, image_size, k=1):
        model = torch.load(self.model_path)
        set_as_feature_extractor(model)
        segmentation_model = torch.load(self.segmentation_model_path)

        image = cv2.resize(
            cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB), (256, 256)
        )

        x = TF.to_tensor(image)
        mask = np.squeeze((segmentation_model(x.unsqueeze(0)) > 0.5).numpy())

        plt.imshow(image)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
        bg = (1 - mask).astype("uint8")
        out = cv2.resize(image * mask + bg * 255, image_size)

        plt.imshow(out)
        plt.show()
        x = TF.to_tensor(out)
        feature = model(x.unsqueeze(0))
        print("Loading features")

        t0 = time.time()
        print("Looking for ...")
        similar = self.tree.query(feature, k=k, return_distance=False)

        print("Found in {}s".format(time.time() - t0))

        result = []
        for i in similar[0]:
            result.append(self.index[i][:-4])

        return result
