import codecs
import errno
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import torch
import pickle
from torchvision import transforms
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from sklearn import preprocessing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ClassificationDataset(Dataset):
    """Dataset that on each iteration provides two random pairs of
    MNIST images. One pair is of the same number (positive sample), one
    is of two different numbers (negative sample).
    """

    def __init__(self, data_path, data_csv, image_size, transform=None,
                 distinguish_class='masterCategory', thr=50,
                 load_path=None, load_in_ram=False):
        if not self._check_exists(data_csv, data_path):
            raise RuntimeError('Dataset not found')
        self.data_path = data_path
        self.image_size = image_size

        df = pd.read_csv(data_csv)
        self.transform = transform

        df = df[
            df.apply(lambda x: (x[distinguish_class] is not None) and os.path.exists(os.path.join(
                data_path, str(x['id']) + '.jpg')),
                     axis=1)]
        df.dropna(inplace=True)

        vc = df[distinguish_class].value_counts()
        u = [i not in set(vc[vc < thr].index) for i in df[distinguish_class]]
        df = df[u]

        print(df[distinguish_class].describe(include=['object']))
        print(df.groupby(distinguish_class)['id'].nunique())
        self.n_classes = len(np.unique(df[distinguish_class]))

        df.reset_index(inplace=True)
        self.images_id = df['id'].values
        self.images = None

        print("Building classification dataset ")
        if load_path:

            with open(os.path.join(load_path, 'images_id.pickle'), 'rb') as pic:
                self.images_id = pickle.load(pic)

            with open(os.path.join(load_path, 'targets.pickle'), 'rb') as pic:
                self.targets = pickle.load(pic)

        else:
            targets = df[distinguish_class].values
            le = preprocessing.LabelEncoder()
            le.fit(targets)
            self.le = le
            self.targets = le.transform(df[distinguish_class].values)

        if load_in_ram:
            self.images = {i: self.load_image_as_tensor(i) for i in self.images_id}

        print("Building dataset -- Done")

    def save(self, save_path):

        with open(os.path.join(save_path, 'images_id.pickle'), 'wb') as pic:
            pickle.dump(self.images_id, pic)

        with open(os.path.join(save_path, 'targets.pickle'), mode='wb') as pic:
            pickle.dump(self.targets, pic)

    def load_image_as_tensor(self, img_id, ext='.jpg'):
        if self.images:
            x = self.images[img_id]
        else:
            image_path = os.path.join(self.data_path, str(img_id) + ext)
            image = Image.open(image_path).convert('RGB').resize(self.image_size)
            x = TF.to_tensor(image)
        return x

    def __getitem__(self, index):

        image = self.load_image_as_tensor(self.images_id[index])
        target = torch.tensor(self.targets[index])

        return image.to(device), target.to(device)

    def __len__(self):
        return len(self.images_id)

    def _check_exists(self, csv_path, data_path):
        return os.path.exists(data_path) and os.path.exists(csv_path)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.data_path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class SiameseDataset(Dataset):
    """Dataset that on each iteration provides two random pairs of
    MNIST images. One pair is of the same number (positive sample), one
    is of two different numbers (negative sample).
    """

    def __init__(self, data_path, data_csv, image_size, transform=None,
                 distinguish_class='masterCategory', thr=50,
                 load_path=None, load_in_ram=False):
        if not self._check_exists(data_csv, data_path):
            raise RuntimeError('Dataset not found')
        self.data_path = data_path
        self.image_size = image_size

        df = pd.read_csv(data_csv)
        self.transform = transform

        df = df[
            df.apply(lambda x: (x[distinguish_class] is not None) and os.path.exists(os.path.join(
                data_path, str(x['id']) + '.jpg')),
                     axis=1)]
        df.dropna(inplace=True)

        vc = df[distinguish_class].value_counts()
        u = [i not in set(vc[vc < thr].index) for i in df[distinguish_class]]
        df = df[u]

        iterative_copy = df.copy()

        print(df[distinguish_class].describe(include=['object']))
        print(df.groupby(distinguish_class)['id'].nunique())

        df.reset_index(inplace=True)
        self.images_id = df['id'].values
        self.images = None
        print("Building siamese dataset ")
        if load_path:
            with open(os.path.join(load_path, 'images_id.pickle'), 'rb') as pic:
                self.images_id = pickle.load(pic)

            with open(os.path.join(load_path, 'positive.pickle'), 'rb') as pic:
                self.positive_images = pickle.load(pic)

            with open(os.path.join(load_path, 'negative.pickle'), 'rb') as pic:
                self.negative_images = pickle.load(pic)
        else:
            self.positive_images = []
            self.negative_images = []

            for x1, y1 in zip(iterative_copy['id'], iterative_copy[distinguish_class]):
                try:
                    positive_index = df.loc[(df[distinguish_class] == y1) & (df['id'] != x1)][
                        'id'].values[0]
                    negative_index = df.loc[df[distinguish_class] != y1]['id'].values[0]
                    self.positive_images.append(positive_index)
                    self.negative_images.append(negative_index)
                    df = df.sample(frac=1)
                except:
                    print("Issue")

        if load_in_ram:
            self.images = {i: self.load_image_as_tensor(i) for i in self.images_id}

        print("Building siamese Done")

    def save(self, save_path):

        with open(os.path.join(save_path, 'images_id.pickle'), 'wb') as pic:
            pickle.dump(self.images_id, pic)

        with open(os.path.join(save_path, 'positive.pickle'), mode='wb') as pic:
            pickle.dump(self.positive_images, pic)

        with open(os.path.join(save_path, 'negative.pickle'), mode='wb') as pic:
            pickle.dump(self.negative_images, pic)

    def load_image_as_tensor(self, img_id, ext='.jpg'):
        if self.images:
            x = self.images[img_id]
        else:
            image_path = os.path.join(self.data_path, str(img_id) + ext)
            image = Image.open(image_path).convert('L').resize(self.image_size)
            x = TF.to_tensor(image)
        return x

    def __getitem__(self, index):

        image1 = self.load_image_as_tensor(self.images_id[index])

        positive_image = self.load_image_as_tensor(self.positive_images[index])
        negative_image = self.load_image_as_tensor(self.negative_images[index])

        positive_example = torch.stack([image1, positive_image])
        negative_example = torch.stack([image1, negative_image])

        target = torch.tensor([1, 0])

        return torch.stack([positive_example, negative_example]), target

    def __len__(self):
        return len(self.images_id)

    def _check_exists(self, csv_path, data_path):
        return os.path.exists(data_path) and os.path.exists(csv_path)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.data_path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
