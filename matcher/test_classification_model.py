import torch
import torch.nn.functional as F
from torchvision import transforms
from matcher.models import ClassificationNet
from matcher.dataset import ClassificationDataset
from torch.utils.data import DataLoader


def main():
    config = {
        "save_every_freq": False,
        "save_frequency": 2,
        "save_best": True,
        "labels": ["masterCategory", "subCategory"],
        "model_name": "resnet18",
        "batch_size": 16,
        "lr": 0.001,
        "num_epochs": 50,
        "weight_decay": 0.0001,
        "exp_base_dir": "data/exps/exp3",
        "image_size": [224, 224],
        "load_path": None,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = ClassificationDataset(
        "./data/images/",
        "./data/small_train.csv",
        distinguish_class=config["classes"],
        load_path=None,
        image_size=config["image_size"],
        transform=normalize,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    val_loader = DataLoader(
        ClassificationDataset(
            "./data/images",
            "./data/small_test.csv",
            distinguish_class=config["classes"],
            image_size=config["image_size"],
            transform=normalize,
            label_encoder=train_dataset.les,
        ),
        batch_size=config["batch_size"],
        shuffle=True,
    )

    model = ClassificationNet(
        image_size=config["image_size"],
        n_classes=train_loader.dataset.n_classes,
        name=config["model_name"],
    ).to(device)

    model.load_state_dict(torch.load(config["load_path"]))
    test(model, device, val_loader, len(config["labels"]))


def test(model, device, test_loader, n_label=3):
    model.eval()

    with torch.no_grad():
        accurate_labels = 0
        all_labels = 0
        val_loss = []
        accurate_labels = [0, 0, 0]
        accuracies = [0, 0, 0]
        for batch_idx, (data, target) in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to(device)

            output = model(data)
            target = target.long()
            val_loss.append(
                [
                    F.cross_entropy(torch.squeeze(output[i]), target[:, i])
                    for i in range(n_label)
                ]
            )

            for i in range(n_label):
                accurate_labels[i] += torch.sum(
                    (torch.argmax(F.softmax(output[i]), dim=1) == target[:, i])
                )

            all_labels += len(target)

        for i in range(n_label):
            accuracies[i] = 100.0 * accurate_labels[i].item() / all_labels
        print(
            "Test accuracy: ({})/{} ({}), Loss: ({})".format(
                ", ".join([str(accurate_labels[i].item()) for i in range(n_label)]),
                all_labels,
                ", ".join(["{:.3f}%".format(accuracies[i]) for i in range(n_label)]),
                ", ".join(
                    "{:.6f}".format(loss)
                    for loss in torch.mean(torch.tensor(val_loss), dim=0).data.tolist()
                ),
            )
        )
        return accuracies


if __name__ == "__main__":
    main()
