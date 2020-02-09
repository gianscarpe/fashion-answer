import os
from PIL import Image
from torch.nn import Sequential
from matcher.features import FeatureMatcher


def get_feature_extractor(model):
    model.pre_net.classifier = Sequential()


if __name__ == "__main__":
    config = {
        "input": "data/examples/11100.jpg",
        "data_path": "data/fashion-product-images-small/images",
        "exp_base_dir": "data/exps/exp1",
        "image_size": [224, 224],
        "phase_1_model": "data/models/resnet18_phase1_best.pt",
        "phase_2_model": "data/models/resnet18_phase2_best.pt",
        "features_path": "data/features/features_resnet18_phase2.npy",
        "index_path": "data/features/features_resnet18_phase2.pickle",
        "segmentation_path": "data/models/segm.pth",
    }
    # ["gender", "masterCategory", "subCatetgory"]

    fm = FeatureMatcher(
        features_path=config["features_path"],
        phase1_params_path=config["phase_1_model"],
        phase2_params_path=config["phase_2_model"],
        image_size=config["image_size"],
        index_path=config["index_path"],
        segmentation_model_path=config["segmentation_path"],
    )

    image = Image.open(config['input']).convert("RGB")
    if False:
        image = fm.segment_image(image)

    master_classe = fm.classify(image, image_size=config['image_size'], phase=1)
    sub_classe = fm.classify(image, image_size=config['image_size'], phase=2)

    print(f"Master: {master_classe} \n sub: {sub_classe}")

    result = fm.get_k_most_similar(
        image, image_size=config['image_size'], k=3, segmentation=False
    )
    for r in result:
        image_path = os.path.join(config['data_path'], str(r))
        Image.open(image_path).show()

    print("Done")
