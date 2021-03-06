import torch
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from matcher.models import ClassificationNet
from matcher.dataset import ClassificationDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import time
import os
import numpy as np


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
        "exp_base_dir": "data/exps",
        "image_size": [224, 224],
        "load_path": None,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_epoch = 1
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = ClassificationDataset(
        "./data/images/",
        "./data/small_train.csv",
        distinguish_class=config["labels"],
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
            "./data/small_val.csv",
            distinguish_class=config["labels"],
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

    if config["load_path"]:
        print("Loading and evaluating model")
        start_epoch = int(config["load_path"][-6:-3])
        model = torch.load(config["load_path"])
        test(model, device, val_loader)

    optimizer = optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    best_accu = 0.0
    for epoch in range(start_epoch, config["num_epochs"] + 1):
        train(
            model,
            device,
            train_loader,
            epoch,
            optimizer,
            config["batch_size"],
            len(config["labels"]),
        )
        accuracies = test(model, device, val_loader, len(config["labels"]))
        if config["save_every_freq"]:
            if epoch % config["save_frequency"] == 0:
                torch.save(
                    model,
                    os.path.join(
                        config["exp_base_dir"],
                        config["model_name"] + "_{:03}.pt".format(epoch),
                    ),
                )
        if config["save_best"]:
            accu = sum(accuracies) / len(config["labels"])
            if accu > best_accu:
                print("* PORCA L'OCA SAVE BEST")
                best_accu = accu
                torch.save(
                    model,
                    os.path.join(
                        config["exp_base_dir"], config["model_name"] + "_best.pt"
                    ),
                )


def train(model, device, train_loader, epoch, optimizer, batch_size, n_label=3):
    model.train()
    t0 = time.time()
    training_loss = []
    criterions = [CrossEntropyLoss() for i in range(n_label)]
    for batch_idx, (data, target) in enumerate(train_loader):
        for i in range(len(data)):
            data[i] = data[i].to(device)

        optimizer.zero_grad()
        output = model(data)
        target = target.long()
        loss = 0
        for i in range(n_label):
            loss = loss + criterions[i](torch.squeeze(output[i]), target[:, i])

        loss.backward()
        # loss_items = []
        # for i in range(n_label):
        #     loss_items.append(loss[i].item())
        #     loss[i].backward()

        training_loss.append(loss.item())
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)] \tBatch Loss: ({})".format(
                    epoch,
                    batch_idx * batch_size,
                    len(train_loader.dataset),
                    100.0 * batch_idx * batch_size / len(train_loader.dataset),
                    "{:.6f}".format(loss.item()),
                )
            )
    print(
        "Train Epoch: {}\t time:{:.3f}s \tMeanLoss: ({})".format(
            epoch, (time.time() - t0), "{:.6f}".format(np.average(training_loss))
        )
    )


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
