import torch
import os
import pickle
from PIL import Image
from tqdm import tqdm
from matcher.models import TwoPhaseNet
import torchvision.transforms.functional as TF
import numpy as np

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "two_phase": True,
        "phase": "1",
        "data_path": "data/images",
        "model_name": "resnet18",
        "n_label": 1,
        "image_size": [224, 224],
        "load_path": "../drive/My Drive/fashion-answer/data/models/resnet18_phase1_best.pt",
        "save_path": "../drive/My Drive/fashion-answer/data/features",
    }

    if config["two_phase"]:
        model = TwoPhaseNet(
            image_size=config["image_size"],
            n_classes_phase1=6,
            n_classes_phase2=43,
            name=config["model_name"],
        )
        model.to(device)
        pretrained_dict = torch.load(config["load_path"])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
    elif not config["two_phase"] and config["load_path"]:
        print("Loading model")
        model = torch.load(config["load_path"], map_location=device)
        model.eval()
        model.set_as_feature_extractor(name=config["model_name"])

    images = os.listdir(config["data_path"])
    print("Extracting features")
    mapping = {}
    features = []
    for i, img_id in enumerate(tqdm(images)):
        mapping[i] = img_id
        img_id = img_id[:-4]
        image_path = os.path.join(config["data_path"], str(img_id) + ".jpg")
        image = Image.open(image_path).convert("RGB").resize(config["image_size"])
        x = TF.to_tensor(image)
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        x = x.to(device)
        feature = model(x.unsqueeze(0))
        features.append(
            [np.array(feature[i].data.tolist()[0]) for i in range(config["n_label"])]
        )

    np.save(
        os.path.join(
            config["save_path"],
            "features_" + config["model_name"] + "_phase" + config["phase"] + ".npy",
        ),
        features,
    )
    with open(
        os.path.join(
            config["save_path"],
            "features_" + config["model_name"] + "_phase" + config["phase"] + ".pickle",
        ),
        "wb",
    ) as f:
        pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done")
