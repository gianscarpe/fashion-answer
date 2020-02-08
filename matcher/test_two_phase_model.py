import torch
import torch.nn.functional as F
from torchvision import transforms
from matcher.models import TwoPhaseNet
from matcher.dataset import ClassificationDataset
from torch.utils.data import DataLoader


def main():
    config = {
        "phase": "1",
        "classes": ["masterCategory"],  # subCategory masterCategory
        "model_name": "resnet18",
        "batch_size": 16,
        "image_size": [224, 224],
        "load_path": "data/models/resnet18_phase1_best.pt",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    test_loader = DataLoader(
        ClassificationDataset(
            "./data/images",
            "./data/small_test.csv",
            distinguish_class=config["classes"],
            image_size=config["image_size"],
            transform=normalize,
            thr=5,
        ),
        batch_size=config["batch_size"],
        shuffle=False,
    )

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

    test(model, device, test_loader, n_label=1)


def test(model, device, test_loader, n_label=3):
    model.eval()
    model.to(device)
    with torch.no_grad():
        accurate_labels = 0
        all_labels = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.long().to(device)

            output = model(data)
            accurate_labels += torch.sum(
                (torch.argmax(F.softmax(output), dim=1) == target[:, 0])
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
