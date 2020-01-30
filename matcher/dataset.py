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


class SiameseDataset(Dataset):
    """Dataset that on each iteration provides two random pairs of
    MNIST images. One pair is of the same number (positive sample), one
    is of two different numbers (negative sample).
    """

    def __init__(self, data_path, data_csv, image_size, transform=None,
                 distinguish_class='masterCategory',
                 load_path=None, load_in_ram=False):
        if not self._check_exists(data_csv, data_path):
            raise RuntimeError('Dataset not found')
        self.data_path = data_path
        self.image_size = image_size

        df = pd.read_csv(data_csv)
        self.transform = transform

        df = df[
            df.apply(lambda x: os.path.exists(os.path.join(data_path, str(x['id']) + '.jpg')),
                     axis=1)]
        df.reset_index(inplace=True)
        self.images_id = df['id']
        self.images = None
        print("Building siamese dataset ")
        if load_path:
            with open(os.path.join(load_path, 'positive.pickle'), 'rb') as pic:
                self.positive_images = pickle.load(pic)

            with open(os.path.join(load_path, 'negative.pickle'), 'rb') as pic:
                self.negative_images = pickle.load(pic)
        else:
            self.positive_images = []
            self.negative_images = []

            for x1, y1 in zip(df['id'], df[distinguish_class]):
                positive_index = df.loc[(df[distinguish_class] == y1) & (df['id'] != x1)].index[0]
                negative_index = df.loc[df[distinguish_class] != y1].index[0]
                try:
                    self.positive_images.append(df['id'].iloc[positive_index])
                    self.negative_images.append(df['id'].iloc[negative_index])
                    df = df.sample(frac=1)
                except:
                    print('Issue with %s'.format(positive_index))

        if load_in_ram:
            self.images = {i: self.load_image_as_tensor(i) for i in self.images_id}

        print("Building siamese Done")

    def save(self, save_path):
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
