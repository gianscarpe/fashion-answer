import os
from matcher.features import FeatureMatcher
from bot.Updater import Updater
from PIL import Image

BOT_TOKEN = "1073943883:AAGomT4w81fVftMWxO2-OnP3ZSGB8e2eaQg"

def fileparts(fn):
    (dirName, fileName) = os.path.split(fn)
    (fileBaseName, fileExtension) = os.path.splitext(fileName)
    return dirName, fileBaseName, fileExtension


def get_handler(fm, image_size, data_path, segmentation):
    def imageHandler(
        bot, message, chat_id, local_filename, k=3, segmentation=segmentation
    ):
        print(local_filename)
        # send message to user
        bot.sendMessage(
            chat_id, "Grazie di averci scelto! Stiamo preparando i tuoi suggerimenti"
        )

        image = Image.open(local_filename).convert("RGB")
        if segmentation:
            image = fm.segment_image(image)
        master_classe = fm.classify(image, image_size=image_size, phase=1)
        sub_classe = fm.classify(image, image_size=image_size, phase=2)

        bot.sendMessage(chat_id, f"Master: {master_classe} \n sub: {sub_classe}")

        result = fm.get_k_most_similar(
            image, image_size=image_size, k=k, segmentation=False
        )
        for r in result:
            image_path = os.path.join(data_path, str(r))
            bot.sendImage(chat_id, image_path, "")

    return imageHandler


if __name__ == "__main__":
    config = {
        "data_path": "data/images",
        "exp_base_dir": "data/exps/exp1",
        "image_size": (224, 224),
        "phase_1_model": "data/models/resnet18_phase1_best.pt",
        "phase_2_model": "data/models/resnet18_phase2_best.pt",
        "features_path": "data/features/features_resnet18_phase2.npy",
        "index_path": "data/features/features_resnet18_phase2.pickle",
        "segmentation_path": "data/models/segm.pth",
    }
    # ["gender", "masterCategory", "subCategory"]

    fm = FeatureMatcher(
        features_path=config["features_path"],
        phase1_params_path=config["phase_1_model"],
        phase2_params_path=config["phase_2_model"],
        image_size=config["image_size"],
        index_path=config["index_path"],
        segmentation_model_path=config["segmentation_path"],
    )

    updater = Updater(BOT_TOKEN)
    updater.setPhotoHandler(
        get_handler(fm, config["image_size"], config["data_path"], segmentation=False)
    )
    updater.start()
