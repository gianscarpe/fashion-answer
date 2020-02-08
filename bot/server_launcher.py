import os, sys, platform, subprocess
from config import BOT_TOKEN
from matcher.features import FeatureMatcher
import torch
from Updater import Updater
from PIL import Image


def fileparts(fn):
    (dirName, fileName) = os.path.split(fn)
    (fileBaseName, fileExtension) = os.path.splitext(fileName)
    return dirName, fileBaseName, fileExtension


def get_handler(fm, image_size, data_path, segmentation):
    def imageHandler(bot, message, chat_id, local_filename, k=3, segmentation=segmentation):
        print(local_filename)
        # send message to user
        bot.sendMessage(chat_id, "Grazie di averci scelto! Stiamo preparando i tuoi suggerimenti")

        image = Image.open(local_filename).convert("RGB")
        if segmentation:
            image = fm.segment_image(image)
        classes = fm.classify(image, image_size=image_size)

        labels = [f"In base al tuo sesso {classes[0]}",
                  f"In base al tipo di abito {classes[1]}",
                  f"In base al sottotipo di abito {classes[2]}",
                  f"Simili al tuo stile"]

        for i in range(0, 4):
            bot.sendMessage(chat_id, labels[i])
            result = fm.get_k_most_similar(image, image_size=image_size,
                                           k=k, similar_type=i)
            for r in result:
                image_path = os.path.join(data_path, str(r))
                bot.sendImage(chat_id, image_path, "")

    return imageHandler


if __name__ == "__main__":
    config = {
        'data_path': 'data/fashion-product-images-small/images',
        'exp_base_dir': 'data/exps/exp1',
        'image_size': (224, 224),
        'load_path': "data/models/alexnet_best.pt",
        'features_path': 'data/features/featuresalexnet.npy',
        'index_path': 'data/features/featuresalexnet.pickle',
        'segmentation_path': 'data/models/segm.pth',

    }
    # ["gender", "masterCategory", "subCategory"]

    fm = FeatureMatcher(features_path=config['features_path'], model_path=config['load_path'],
                        index_path=config['index_path'],
                        segmentation_model_path=config['segmentation_path'])

    updater = Updater(BOT_TOKEN)
    updater.setPhotoHandler(get_handler(fm, config['image_size'], config['data_path'],
                                        segmentation=False))
    updater.start()
