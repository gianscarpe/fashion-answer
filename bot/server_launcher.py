import os
from matcher.features import FeatureMatcher
from bot.Updater import Updater
from PIL import Image
from matcher.dataset import ClassificationDataset


BOT_TOKEN = "1073943883:AAGomT4w81fVftMWxO2-OnP3ZSGB8e2eaQg"


def fileparts(fn):
    (dirName, fileName) = os.path.split(fn)
    (fileBaseName, fileExtension) = os.path.splitext(fileName)
    return dirName, fileBaseName, fileExtension


def get_handler(
    fm,
    image_size,
    data_path,
    segmentation,
    label_encoder_master=None,
    label_encoder_sub=None,
):
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

        bot.sendMessage(
            chat_id,
            f"Master category:\t{label_encoder_master.classes_[master_classe]} \nSub category:\t{label_encoder_sub.classes_[sub_classe]}",
        )

        result = fm.get_k_most_similar(
            image, image_size=image_size, k=k, segmentation=segmentation
        )
        for r in result:
            image_path = os.path.join(data_path, str(r))
        bot.sendImage(chat_id, image_path, "")

    return imageHandler


if __name__ == "__main__":
    config = {
        "data_path": "data/fashion-product-images-small/images",
        "exp_base_dir": "data/exps/exp1",
        "image_size": [224, 224],
        "phase_1_model": "data/models/resnet18_phase1_best.pt",
        "phase_2_model": "data/models/resnet18_phase2_best.pt",
        "features_path": "data/features/features_resnet18_phase2.npy",
        "index_path": "data/features/features_resnet18_phase2.pickle",
        "segmentation_path": "data/models/segm.pth",
        "les_path": "data/features/le.pickle"
    }
    # ["gender", "masterCategory", "subCatetgory"]
    import pickle

    with open(config['les_path'], 'rb') as pi:
        les = pickle.load(pi)

    fm = FeatureMatcher(
        features_path=config["features_path"],
        phase1_params_path=config["phase_1_model"],
        phase2_params_path=config["phase_2_model"],
        image_size=config["image_size"],
        index_path=config["index_path"],
        segmentation_model_path=config["segmentation_path"],
    )
    label_encoder_master = ClassificationDataset(
        "./data/images/",
        "./data/small_train.csv",
        distinguish_class=["masterCategory"],
        load_path=None,
        image_size=None,
        transform=None,
    ).les[0]
    label_encoder_sub = ClassificationDataset(
        "./data/images/",
        "./data/small_train.csv",
        distinguish_class=["subCategory"],
        load_path=None,
        image_size=None,
        transform=None,
    ).les[0]
    updater = Updater(BOT_TOKEN)
    updater.setPhotoHandler(
        get_handler(
            fm,
            config["image_size"],
            config["data_path"],
            segmentation=True,
            label_encoder_master=label_encoder_master,
            label_encoder_sub=label_encoder_sub,
        )
    )
    updater.start()
