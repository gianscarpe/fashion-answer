from Updater import Updater
import os, sys, platform, subprocess
from config import BOT_TOKEN
from matcher.features import FeatureMatcher


def fileparts(fn):
    (dirName, fileName) = os.path.split(fn)
    (fileBaseName, fileExtension) = os.path.splitext(fileName)
    return dirName, fileBaseName, fileExtension


def get_handler(fm, image_size, data_path):
    def imageHandler(bot, message, chat_id, local_filename, k=3):
        print(local_filename)
        # send message to user
        bot.sendMessage(chat_id, "Hi, please wait until the image is ready")

        result = fm.get_k_most_similar(local_filename, image_size=image_size, k=k)
        for r in result:
            image_path = os.path.join(data_path, str(r) + ".jpg")
            bot.sendImage(chat_id, image_path, "")

    return imageHandler


if __name__ == "__main__":
    bot_id = BOT_TOKEN
    config = {
        'data_path': 'data/fashion-product-images-small/images',
        'exp_base_dir': 'data/exps/exp1',
        'image_size': (224, 224),
        'load_path': "data/exps/exp1/classification_001.pt",
        'features_path': 'data/fashion-product-images-small/features/features.npy',
        'index_path': 'data/fashion-product-images-small/features/index.pickle',
        'segmentation_path': 'data/models/segm.pth'
    }

    fm = FeatureMatcher(features_path=config['features_path'], model_path=config['load_path'],
                        index_path=config['index_path'],
                        segmentation_model_path=config['segmentation_path'])

    updater = Updater(bot_id)
    updater.setPhotoHandler(get_handler(fm, config['image_size'], config['data_path']))
    updater.start()
