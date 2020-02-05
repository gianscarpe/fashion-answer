import torch
import os
from PIL import Image
import torchvision.transforms.functional as TF
from torch.nn import Sequential
import numpy as np
from sklearn.neighbors import KDTree
import time
import pickle

def get_feature_extractor(model):
    model.pre_net.classifier = Sequential()


if __name__ == '__main__':
    config = {
        'input_path': 'data/images.jpg',
        'data_path': 'data/fashion-product-images-small/images',
        'exp_base_dir': 'data/exps/exp1',
        'image_size': [224, 224],
        'load_path': "data/exps/exp1/classification_001.pt",
        'features_path': 'data/fashion-product-images-small/features/features.npy',
        'index_path': 'data/fashion-product-images-small/features/index.pickle',
    }

    with open(config['index_path'], "rb") as pic:
        index = pickle.load(pic)

    print("Loading model")
    model = torch.load(config['load_path'])
    get_feature_extractor(model)

    image = Image.open(config['input_path']).convert('RGB').resize(config['image_size'])
    image.show()

    x = TF.to_tensor(image)
    feature = model(x.unsqueeze(0))
    print('Loading features')
    t0 = time.time()

    features_matrix = np.squeeze(np.load(config['features_path']))
    tree = KDTree(features_matrix)
    print("Loaded in {}s".format(time.time() - t0))

    t0 = time.time()
    print("Looking for ...")
    similar = tree.query(feature, return_distance=False)

    print("Found in {}s".format(t0 - time.time()))

    img_id = index[similar[0][0]][:-4]
    image_path = os.path.join(config['data_path'], str(img_id) + '.jpg')
    image = Image.open(image_path).convert('RGB').resize(config['image_size'])
    image.show()

    print('Done')
