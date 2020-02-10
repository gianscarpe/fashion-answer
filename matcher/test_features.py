#!/usr/bin/env python
import torch
from torchvision import transforms
import pandas as pd
from matcher.dataset import ClassificationDataset
from matcher.features import FeatureMatcher
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
config = {
    "data_path": "data/fashion-product-images-small/images",
    "exp_base_dir": "data/exps/exp1",
    "image_size": [224, 224],
    "phase_1_model": "data/models/resnet18_phase1_best.pt",
    "phase_2_model": "data/models/resnet18_phase2_best.pt",
    "features_path": "data/features/features_resnet18_phase2_new.npy",
    "index_path": "data/features/features_resnet18_phase2_new.pickle",
    "segmentation_path": "data/models/segm.pth",
    "les_path": "data/features/le.pickle",
    "classes": ["subCategory"]
}

# ["gender", "subCategory", "masterCategory"]

print("Loading")
dataset = pd.read_csv("data/csv/styles.csv", error_bad_lines=False)
with open(config['les_path'], 'rb') as pi:
    les = pickle.load(pi)

train_dataset = ClassificationDataset(
    config['data_path'],
    "./data/csv/small_train.csv",
    distinguish_class=config["classes"],
    image_size=config["image_size"],
    transform=transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
    thr=5,
)

test_loader = ClassificationDataset(
    config['data_path'],
    "./data/csv/small_test.csv",
    distinguish_class=config["classes"],
    image_size=config["image_size"],
    transform=transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
    thr=5,
    label_encoder=[les[1]],
)

# %%
fm = FeatureMatcher(
    features_path=config["features_path"],
    phase1_params_path=config["phase_1_model"],
    phase2_params_path=config["phase_2_model"],
    image_size=config["image_size"],
    index_path=config["index_path"],
    segmentation_model_path=config["segmentation_path"],
)

# %%
accurate_labels = [0, 0, 0]
accuracies = [0, 0, 0]
classes = config['classes']

lenloader = len(test_loader)
with torch.no_grad():
    all_labels = 0
    val_loss = []
    for ind in range(0, lenloader):
        image, target = test_loader[ind]
        try:
            for i in range(0, 1):
                output = fm.get_k_most_similar(
                    image, image_size=config["image_size"]
                )

                key = int(output[0][:-4])
                target_class = les[1].inverse_transform(
                    [int(target[0])]
                )[0]

                found = dataset[dataset["id"] == key][classes[i]].values[0]
                accurate_labels[i] += found == target_class

        except:
            print("Missing")

        print(f"Immagine {ind}/{lenloader}")
        print(
            f"Temp. accuracy {accurate_labels[0] / (ind + 1)} \n {accurate_labels[1] / (ind + 1)}")

print(f"Label accurate {accurate_labels}")

n_label = len(classes)
for i in range(n_label):
    accuracies[i] = 100.0 * accurate_labels[i] / lenloader
    print(
        "Test accuracy: ({})/{} ({})".format(
            accurate_labels[i], lenloader, accuracies[i]
        )
    )
