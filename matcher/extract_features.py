import torch
import os
from PIL import Image
import torchvision.transforms.functional as TF
from torch.nn import Sequential
import numpy as np


def get_feature_extractor(model):
    model.pre_net.classifier = Sequential()


if __name__ == '__main__':

    config = {
        'data_path': 'data/fashion-product-images-small/images',
        'exp_base_dir': 'data/exps/exp1',
        'image_size': [224, 224],
        'load_path': "data/exps/exp1/classification_001.pt",
        'save_path': 'data/fashion-product-images-small/features',
    }

    if config['load_path']:
        print("Loading model")
        model = torch.load(config['load_path'])
        get_feature_extractor(model)

    images = os.listdir(config['data_path'])
    print('Extracting features')
    features = []
    for img_id in images:
        img_id = img_id[:-4]
        image_path = os.path.join(config['data_path'], str(img_id) + '.jpg')
        image = Image.open(image_path).convert('RGB').resize(config['image_size'])
        x = TF.to_tensor(image)
        feature = model(x.unsqueeze(0))
        features.append(feature.numpy())

    np.save(os.path.join(config['save_path'], 'features.npy'), features)
    print('Done')
