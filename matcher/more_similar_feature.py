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
        'input_path': 'data/images.jpg',
        'data_path': 'data/fashion-product-images-small/images',
        'exp_base_dir': 'data/exps/exp1',
        'image_size': [224, 224],
        'load_path': "data/exps/exp1/classification_001.pt",
        'features_path': 'data/fashion-product-images-small/features',
    }

    print("Loading model")
    model = torch.load(config['load_path'])
    get_feature_extractor(model)

    image = Image.open(config['input_path']).convert('RGB').resize(config['image_size'])
    image.show()

    x = TF.to_tensor(image)
    feature = model(x.unsqueeze(0))
    features_path = os.listdir(config['features_path'])
    print('Extracting features')

    min_sim = 999
    max_id = 0
    for feature_id in features_path:
        feature_path = os.path.join(config['features_path'], feature_id)
        ft = np.load(feature_path)

        sim = np.linalg.norm(feature - ft)
        if sim < min_sim:
            min_sim = sim
            max_id = feature_id

    print('More similar feature is {}'.format(max_id))

    img_id = max_id[:-4]
    image_path = os.path.join(config['data_path'], str(img_id) + '.jpg')
    image = Image.open(image_path).convert('RGB').resize(config['image_size'])
    image.show()

    print('Done')
