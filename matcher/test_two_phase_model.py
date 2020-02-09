import torch
import torch.nn.functional as F
from torchvision import transforms
from matcher.models import TwoPhaseNet
from matcher.dataset import ClassificationDataset
from torch.utils.data import DataLoader
from matcher.features import FeatureMatcher


def main():
    config = {
        "phase": "1",
        "classes": ["masterCategory", 'subCategory'],  # subCategory masterCategory
        "model_name": "resnet18",
        "batch_size": 16,
        "image_size": [224, 224],
        "load_path": "data/models/resnet18_phase1_best.pt",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = ClassificationDataset(
        "./data/fashion-product-images-small/images",
        "./data/csv/small_train.csv",
        distinguish_class=config["classes"],
        image_size=config["image_size"],
        transform=normalize,
        thr=5,
    )

    test_loader = DataLoader(
        ClassificationDataset(
            "./data/fashion-product-images-small/images",
            "./data/csv/small_test.csv",
            distinguish_class=config["classes"],
            image_size=config["image_size"],
            transform=normalize,
            thr=5,
            label_encoder=train_dataset.les
        ),
        batch_size=1,
        shuffle=False,
    )

    import pickle
    with open("le.pickle", 'wb') as pi:
        pickle.dump(train_dataset.les, pi)

    model = TwoPhaseNet(
        image_size=config["image_size"],
        n_classes_phase1=6,
        n_classes_phase2=43,
        name=config["model_name"],
    )

    if config["phase"] == "1":
        model.phase1()
    elif config["phase"] == "2":
        model.phase2()

    if config["load_path"]:
        model.load_state_dict(torch.load(config["load_path"], map_location=device))

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

    fm = FeatureMatcher(
        features_path=config["features_path"],
        phase1_params_path=config["phase_1_model"],
        phase2_params_path=config["phase_2_model"],
        image_size=config["image_size"],
        index_path=config["index_path"],
        segmentation_model_path=config["segmentation_path"],
    )

    test(fm.phase1, fm, device, test_loader, n_label=1)


def test(model, fm, device, test_loader, n_label=3):
    model.eval()
    model.to(device)
    with torch.no_grad():
        accurate_labels = 0
        all_labels = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.long().to(device)

            output = fm.classify(data[0], image_size=[224, 224])
            accurate_labels += torch.sum(
                (output == target[:, 0])
            )

            all_labels += len(target)

        accuracy = 100.0 * accurate_labels.item() / all_labels
        print(
            "Test accuracy: ({})/{} ({})".format(
                str(accurate_labels.item()), all_labels, "{:.3f}%".format(accuracy)
            )
        )
        return accuracy


if __name__ == "__main__":
    main()
