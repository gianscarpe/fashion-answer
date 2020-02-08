import pickle
import torch
from PIL import Image
import time
from sklearn.neighbors import KDTree
import numpy as np
import torchvision.transforms.functional as TF
import cv2
from matplotlib import pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.neighbors import DistanceMetric
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling


class FeatureMatcher:

    def __init__(self, model_path, features_path, index_path, segmentation_model_path,
                 segmentation_model_name="efficientnet-b2"):
        with open(index_path, "rb") as pic:
            self.index = pickle.load(pic)

        print("Loading model")
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(segmentation_model_name,
                                                                  "imagenet")

        self.model_path = model_path
        self.segmentation_model_path = segmentation_model_path

        features = np.load(features_path)

        t0 = time.time()

        self.f1 = features[:, 0, :]
        self.f2 = features[:, 1, :]
        self.f3 = features[:, 2, :]

        input_size = self.f1.shape[-1]
        output_size = 256
        self.mcb = CompactBilinearPooling(input_size, input_size, output_size)
        x = torch.tensor(self.f2)
        y = torch.tensor(self.f3)

        self.tot = self.mcb(x, y).numpy()

        self.trees = [KDTree(self.f1), KDTree(self.f2),
                      KDTree(self.f3), KDTree(self.tot)]
        print("Loaded in {}s".format(time.time() - t0))

    def classify(self, image, image_size, device="cpu", segmentation=True):
        model = torch.load(self.model_path, map_location=torch.device(device))

        image = image.resize(image_size)
        image.show()

        x = TF.to_tensor(image)
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        result = model(x.unsqueeze(0))
        return torch.argmax(result[0]), torch.argmax(result[1]), torch.argmax(result[2])

    def segment_image(self, image, device="cpu"):

        image = cv2.resize(np.asarray(image), (256, 256))
        segmentation_model = torch.load(self.segmentation_model_path, map_location=torch.device(
            device))

        x = TF.to_tensor(image).type(torch.FloatTensor)

        mask = np.squeeze((segmentation_model(x.unsqueeze(0)) > 0.5).numpy())

        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
        bg = (1 - mask).astype("uint8")
        image = Image.fromarray(image * mask + bg * 255)
        image.show()
        return image

    def get_k_most_similar(self, image, image_size, k=1, device="cpu", similar_type=0,
                           net_name="resnet"):

        model = torch.load(self.model_path, map_location=torch.device(device))

        image = image.resize(image_size)

        x = TF.to_tensor(image)
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        model.set_as_feature_extractor(name=net_name)
        feature = model(x.unsqueeze(0))

        if similar_type < 3:
            feature = feature[similar_type].detach().numpy()
        else:
            feature[1].detach()
            feature[2].detach()

            feature = self.mcb(feature[1], feature[2]).detach().numpy()

        print('Loading features')

        t0 = time.time()
        print("Looking for ...")
        similar = self.trees[similar_type].query(feature, k=k, return_distance=False)

        print("Found in {}s".format(time.time() - t0))

        result = []
        for i in similar[0]:
            result.append(self.index[i])

        return result
