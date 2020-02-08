# %%
import torch
from matcher.features import FeatureMatcher
from matcher.dataset import ClassificationDataset
from torchvision import transforms
import pandas as pd

config = {
    "data_path": "data/fashion-product-images-small/images",
    "exp_base_dir": "data/exps/exp1",
    "image_size": (224, 224),
    "load_path": "data/models/alexnet_best.pt",
    "features_path": "data/features/featuresalexnet.npy",
    "index_path": "data/features/featuresalexnet.pickle",
    "segmentation_path": "data/models/segm.pth",
    "classes": ["masterCategory", "subCategory", "gender"],
    "segmentation": False,
}
# ["gender", "subCategory", "masterCategory"]

# %%
fm = FeatureMatcher(
    features_path=config["features_path"],
    model_path=config["load_path"],
    index_path=config["index_path"],
    segmentation_model_path=config["segmentation_path"],
)

# %%
print("Loading")
dataset = pd.read_csv("./data/styles.csv", error_bad_lines=False)

test_loader = ClassificationDataset(
    "./data/fashion-product-images-small/images",
    "./data/small_test.csv",
    distinguish_class=config["classes"],
    image_size=config["image_size"],
    transform=transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
    thr=5,
)

# %%
classes = ["masterCategory", "subCategory", "gender"]
device = "cpu"
accurate_labels = [0, 0, 0]
accuracies = [0, 0, 0]
lenloader = len(test_loader)
with torch.no_grad():
    all_labels = 0
    val_loss = []
    for ind in range(lenloader):
        image, target = test_loader[ind]
        for similarity in range(0, 2):  # Per ogni classe di similarità
            if config["segmentation"]:
                image = fm.segment_image(image)
            try:
                output = fm.get_k_most_similar(
                    image, similar_type=similarity, image_size=config["image_size"]
                )
                key = int(output[0][:-4])
                target_class = test_loader.les[similarity].inverse_transform(
                    [int(target[similarity])]
                )[0]
                found = dataset[dataset["id"] == key][classes[similarity]].values[0]
                accurate_labels[similarity] += found == target_class

            except:
                print("Missing")

        print(f"Immagine {ind}/{lenloader}")

print(f"Label accurate {accurate_labels}")

n_label = len(classes)
for i in range(n_label):
    accuracies[i] = 100.0 * accurate_labels[i] / lenloader
    print(
        "Test accuracy: ({})/{} ({})".format(
            accurate_labels[i], lenloader, accuracies[i]
        )
    )
