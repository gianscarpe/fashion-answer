import torch
import os
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "data_path": "data/images",
        "image_size": [224, 224],
        "load_path": "data/exps/exp2/resnet18_best.pt",
        "save_path": "data/features",
    }

    if config["load_path"]:
        print("Loading model")
        model = torch.load(config["load_path"], map_location=device)
        model.set_as_feature_extractor(name="resnet18")
        # get_feature_extractor(model)

    images = os.listdir(config["data_path"])
    print("Extracting features")
    features = []
    for img_id in images:
        img_id = img_id[:-4]
        image_path = os.path.join(config["data_path"], str(img_id) + ".jpg")
        image = Image.open(image_path).convert("RGB").resize(config["image_size"])
        x = TF.to_tensor(image)
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        feature = model(x.unsqueeze(0))
        features.append(feature.numpy())

    np.save(os.path.join(config["save_path"], "features.npy"), features)
    print("Done")
