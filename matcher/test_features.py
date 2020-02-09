#!/usr/bin/env python
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
import os
from PIL import Image
import pickle
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from sklearn import preprocessing
import time
from sklearn.neighbors import KDTree
import segmentation_models_pytorch as smp
from compact_bilinear_pooling import CompactBilinearPooling
from torch import nn
from torchvision import models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClassificationDataset(Dataset):
    """Dataset that on each iteration provides two random pairs of
    MNIST images. One pair is of the same number (positive sample), one
    is of two different numbers (negative sample).
    """

    def __init__(
        self,
        data_path,
        data_csv,
        image_size,
        transform=None,
        distinguish_class="masterCategory",
        thr=50,
        load_path=None,
        load_in_ram=False,
        label_encoder=None,
    ):
        if not self._check_exists(data_csv, data_path):
            raise RuntimeError("Dataset not found")
        self.data_path = data_path
        self.image_size = image_size

        df = pd.read_csv(data_csv)
        self.transform = transform
        if label_encoder:
            df = df[
                df.apply(
                    lambda x: all(
                        x[distinguish_cls] is not None
                        and x[distinguish_cls] in label_encoder[col].classes_
                        for col, distinguish_cls in enumerate(distinguish_class)
                    )
                    and os.path.exists(os.path.join(data_path, str(x["id"]) + ".jpg")),
                    axis=1,
                )
            ]
        else:
            df = df[
                df.apply(
                    lambda x: all(
                        x[distinguish_cls] is not None
                        for distinguish_cls in distinguish_class
                    )
                    and os.path.exists(os.path.join(data_path, str(x["id"]) + ".jpg")),
                    axis=1,
                )
            ]
        df.dropna(inplace=True)
        print(df[distinguish_class])
        # vc = df[distinguish_class].value_counts()
        # u = [i not in set(vc[vc < thr].index) for i in df[distinguish_class]]
        # df = df[u]

        print(df[distinguish_class].describe(include=["object"]))
        print(df.groupby(distinguish_class)["id"].nunique())

        df.reset_index(inplace=True)
        self.images_id = df["id"].values
        self.images = None
        self.les = []

        print("Building classification dataset ")
        if load_path:

            with open(os.path.join(load_path, "images_id.pickle"), "rb") as pic:
                self.images_id = pickle.load(pic)

            with open(os.path.join(load_path, "targets.pickle"), "rb") as pic:
                self.targets = pickle.load(pic)

        else:
            print(df[distinguish_class].shape)
            self.targets = np.zeros(df[distinguish_class].shape)
            self.n_classes = []
            if label_encoder:
                self.les = label_encoder
            for col, distinguish_cls in enumerate(distinguish_class):
                targets = df[distinguish_cls].values
                self.n_classes.append(len(np.unique(targets)))
                if not label_encoder:
                    le = preprocessing.LabelEncoder()
                    le.fit(targets)
                    self.targets[:, col] = le.transform(targets)
                    self.les.append(le)
                else:
                    self.targets[:, col] = self.les[col].transform(targets)

        print(self.targets, self.n_classes)

        if load_in_ram:
            self.images = {i: self.load_image_as_tensor(i) for i in self.images_id}

        print("Building dataset -- Done")

    def save(self, save_path):

        with open(os.path.join(save_path, "images_id.pickle"), "wb") as pic:
            pickle.dump(self.images_id, pic)

        with open(os.path.join(save_path, "targets.pickle"), mode="wb") as pic:
            pickle.dump(self.targets, pic)

    def load_image_as_tensor(self, img_id, ext=".jpg"):
        if self.images:
            x = self.images[img_id]
        else:
            image_path = os.path.join(self.data_path, str(img_id) + ext)
            image = Image.open(image_path).convert("RGB").resize(self.image_size)
            x = TF.to_tensor(image)
        return x

    def __getitem__(self, index):

        image = self.load_image_as_tensor(self.images_id[index])
        if self.transform:
            image = self.transform(image)
        target = torch.tensor(self.targets[index])

        return image.to(device), target.to(device)

    def __len__(self):
        return len(self.images_id)

    def _check_exists(self, csv_path, data_path):
        return os.path.exists(data_path) and os.path.exists(csv_path)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.data_path)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class FeatureMatcher:
    def __init__(
        self,
        model_path,
        features_path,
        index_path,
        segmentation_model_path,
        segmentation_model_name="efficientnet-b2",
        device="cpu",
    ):
        with open(index_path, "rb") as pic:
            self.index = pickle.load(pic)

        print("Loading model")
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            segmentation_model_name, "imagenet"
        )

        self.model_path = model_path
        self.segmentation_model_path = segmentation_model_path
        self.model = torch.load(self.model_path, map_location=torch.device(device))

        self.model = TwoPhaseNet(
            image_size=config["image_size"],
            n_classes_phase1=6,
            n_classes_phase2=43,
            name="resnet18",
        )
        self.model.to(device)
        pretrained_dict = torch.load(
            config["load_path"], map_location=torch.device("cpu")
        )
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(pretrained_dict)

        # self.segmentation_model = torch.load(
        #     self.segmentation_model_path, map_location=torch.device(device)
        # )
        features = np.load(features_path)

        t0 = time.time()

        self.f1 = np.squeeze(features)
        input_size = self.f1.shape[-1]
        output_size = input_size
        print(self.f1.shape)
        self.mcb = CompactBilinearPooling(input_size, input_size, output_size)

        self.trees = [KDTree(self.f1)]
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

    def extract_feature(self, x, image_size, similar_type):
        if type(x) != torch.Tensor:
            x = x.resize(image_size)
            x = TF.to_tensor(x)
            x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        feature = self.model(x.unsqueeze(0))

        return feature

    def get_k_most_similar(
        self, x, image_size, k=1, device="cpu", similar_type=0, net_name="resnet"
    ):

        print("Loading features")

        feature = self.extract_feature(x, image_size, similar_type)
        t0 = time.time()
        print("Looking for ...")
        similar = self.trees[similar_type].query(feature, k=k, return_distance=False)

        print("Found in {}s".format(time.time() - t0))

        result = []
        for i in similar[0]:
            result.append(self.index[i])

        return result


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class TwoPhaseNet(nn.Module):
    def __init__(
        self, image_size, n_classes_phase1, n_classes_phase2, phase="1", name="resnet34"
    ):
        super(TwoPhaseNet, self).__init__()

        self.image_size = image_size
        self.n_classes_phase1 = n_classes_phase1
        self.n_classes_phase2 = n_classes_phase2
        self.phase = phase
        resnet_num = name.split("resnet")[1]
        if resnet_num != "18" and resnet_num != "34":
            raise Exception("Only resnet18 and resnet34 are supported")
        self.name = name
        self.pre_net = getattr(models, name)(pretrained=True)
        self.pre_net.fc = Identity()
        self.features = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
        )
        self.classifier = Identity()

    def phase1(self):
        for child in self.pre_net.children():
            for param in child.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.n_classes_phase1),
        )

    def phase2(self):
        for child in self.pre_net.children():
            for param in child.parameters():
                param.requires_grad = True
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.n_classes_phase2),
        )

    def forward(self, x):
        x = self.pre_net(x)
        x = self.features(x)
        x = self.classifier(x)
        return x


# %%
config = {
    "data_path": "./data/images",
    "image_size": (224, 224),
    "load_path": "./data/models/resnet18_phase2_best.pt",
    "features_path": "./data/features/features_resnet18_phase2.npy",
    "index_path": "./data/features/features_resnet18_phase2.pickle",
    "segmentation_path": None,
    "classes": ["masterCategory"],
    "segmentation": False,
}
# ["gender", "subCategory", "masterCategory"]

print("Loading")
dataset = pd.read_csv("data/styles.csv", error_bad_lines=False)

train_dataset = test_loader = ClassificationDataset(
    "./data/images",
    "./data/small_train.csv",
    distinguish_class=config["classes"],
    image_size=config["image_size"],
    transform=transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
    thr=5,
)

test_loader = ClassificationDataset(
    "./data/images",
    "./data/small_test.csv",
    distinguish_class=config["classes"],
    image_size=config["image_size"],
    transform=transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
    thr=5,
    label_encoder=train_dataset.les,
)

# %%
fm = FeatureMatcher(
    features_path=config["features_path"],
    model_path=config["load_path"],
    index_path=config["index_path"],
    segmentation_model_path=config["segmentation_path"],
)

# %%
accurate_labels = [0, 0, 0]
accuracies = [0, 0, 0]

classes = ["masterCategory"]
device = "cpu"

lenloader = len(test_loader)
with torch.no_grad():
    all_labels = 0
    val_loss = []
    for ind in range(0, lenloader):
        image, target = test_loader[ind]
        for similarity in range(0, 1):  # Per ogni classe di similarità
            try:
                output = fm.get_k_most_similar(
                    image, similar_type=similarity, image_size=config["image_size"]
                )

                key = int(output[0][:-4])
                target_class = test_loader.les[similarity].inverse_transform(
                    [int(target[similarity])]
                )[0]
                found = dataset[dataset["id"] == key][classes[similarity]].values[0]
                accurate_labels[similarity] += found == target_class
            except:
                print("Missing")

        print(f"Immagine {ind}/{lenloader}")
        print(f"Temp. accuracy {accurate_labels[0]/(ind+1)}")

print(f"Label accurate {accurate_labels}")

n_label = len(classes)
for i in range(n_label):
    accuracies[i] = 100.0 * accurate_labels[i] / lenloader
    print(
        "Test accuracy: ({})/{} ({})".format(
            accurate_labels[i], lenloader, accuracies[i]
        )
    )
