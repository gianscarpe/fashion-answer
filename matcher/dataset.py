import numpy as np
import os
from PIL import Image
import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from sklearn import preprocessing

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
        label_encoder=None
    ):
        if not self._check_exists(data_csv, data_path):
            raise RuntimeError("Dataset not found")
        self.data_path = data_path
        self.image_size = image_size

        df = pd.read_csv(data_csv)
        self.transform = transform

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
