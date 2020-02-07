import torch
import os
from PIL import Image
import torchvision.transforms.functional as TF
from torch.nn import Sequential
import numpy as np
from sklearn.neighbors import KDTree
import time
import pickle
from matcher.features import FeatureMatcher


def get_feature_extractor(model):
    model.pre_net.classifier = Sequential()


if __name__ == '__main__':
    config = {
        'input_path': 'data/input.jpg',
        'data_path': 'data/fashion-product-images-small/images',
        'exp_base_dir': 'data/exps/exp1',
        'image_size': (224, 224),
        'load_path': "data/models/resnet18_best.pt",
        'features_path': 'data/features/featuresresnet18.npy',
        'index_path': 'data/features/featuresresnet18_index.pickle',
        'segmentation_path': 'data/models/segm.pth'
    }
    # ["masterCategory", "subCategory", "gender"]

    fm = FeatureMatcher(features_path=config['features_path'], model_path=config['load_path'],
                        index_path=config['index_path'],
                        segmentation_model_path=config['segmentation_path'])

    img_id = fm.get_k_most_similar(config['input_path'], image_size=config['image_size'],
                                   k=3, similar_type=1)[0]

    image_path = os.path.join(config['data_path'], str(img_id))

    image = Image.open(image_path).convert('RGB').resize(config['image_size'])
    image.show()

    print('Done')