import pickle
import torch
from PIL import Image
import time
from sklearn.neighbors import KDTree
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2
import segmentation_models_pytorch as smp
from matcher.models import TwoPhaseNet, Identity


class FeatureMatcher:
    def __init__(
            self,
            phase1_params_path,
            phase2_params_path,
            features_path,
            index_path,
            image_size,
            segmentation_model_path,
            segmentation_model_name="efficientnet-b2",
            device="cpu",
            segmentation=True
    ):

        with open(index_path, "rb") as pic:
            self.index = pickle.load(pic)

        print("Loading model")
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            segmentation_model_name, "imagenet"
        )

        self.segmentation_model_path = segmentation_model_path

        self.image_size = image_size

        # Load phase1

        self.phase1 = TwoPhaseNet(
            image_size=image_size,
            n_classes_phase1=6,
            n_classes_phase2=43,
            name="resnet18",
        )
        self.phase1.phase1()
        self.phase1.load_state_dict(
            torch.load(phase1_params_path, map_location=torch.device("cpu"))
        )
        print(self.phase1)

        # Load phase2
        self.phase2 = TwoPhaseNet(
            image_size=image_size,
            n_classes_phase1=6,
            n_classes_phase2=43,
            name="resnet18",
        )
        self.phase2.phase2()
        self.phase2.load_state_dict(
            torch.load(phase2_params_path, map_location=torch.device("cpu"))
        )
        print(self.phase2)

        # Load phase2 as feature extractor

        self.feature_extractor = TwoPhaseNet(
            image_size=image_size,
            n_classes_phase1=6,
            n_classes_phase2=43,
            name="resnet18",
        )


        self.phase1.eval()
        self.phase2.eval()
        self.feature_extractor.eval()

        self.feature_extractor.to("cpu")
        pretrained_dict = torch.load(phase2_params_path, map_location=torch.device('cpu'))
        model_dict = self.feature_extractor.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.feature_extractor.load_state_dict(pretrained_dict)

        if segmentation is True:
            self.segmentation_model = torch.load(
                self.segmentation_model_path, map_location=torch.device(device)
            )

        features = np.load(features_path)
        print(features.shape)
        t0 = time.time()

        self.features = np.squeeze(features)
        print(f"Features shape {self.features.shape}")
        self.tree = KDTree(self.features)

        print("Loaded in {}s".format(time.time() - t0))

    def classify(self, x, image_size, phase=1):

        self.phase1.to("cpu")
        self.phase2.to("cpu")

        if type(x) != torch.Tensor:
            x = x.resize(image_size)
            x = TF.to_tensor(x)
            x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            for i in range(len(x)):
                x[i] = x[i].to("cpu")

        with torch.no_grad():
            if phase == 1:
                result = self.phase1(x[None])
            elif phase == 2:
                result = self.phase2(x.unsqueeze(0))
            else:
                raise NotImplementedError()
        print("ARGMAX", F.softmax(result))
        return torch.argmax(F.softmax(result), dim=1)

    def segment_image(self, image):

        if type(image) != torch.FloatTensor:
            image = cv2.resize(np.asarray(image), (256, 256))
            x = TF.to_tensor(image).type(torch.FloatTensor)
        else:
            x = image
            image = x.numpy()

        mask = np.squeeze((self.segmentation_model(x.unsqueeze(0)) > 0.5).numpy())

        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
        bg = (1 - mask).astype("uint8")
        image = Image.fromarray(image * mask + bg * 255)
        image.show()
        return image

    def extract_feature(self, x, image_size):
        if type(x) != torch.Tensor:
            x = x.resize(image_size)
            x = TF.to_tensor(x)
            x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        with torch.no_grad():
            feature = self.feature_extractor(x.unsqueeze(0))

        return feature

    def get_k_most_similar(self, x, image_size, k=1, segmentation=False):

        print("Loading features")
        if segmentation:
            x = self.segment_image(x)

        feature = self.extract_feature(x, image_size)
        t0 = time.time()
        print("Looking for ...")
        similar = self.tree.query(feature, k=k, return_distance=False)

        print("Found in {}s".format(time.time() - t0))

        result = []
        for i in similar[0]:
            result.append(self.index[i])

        return result
