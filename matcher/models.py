import torch
from torch import nn
from torch.nn import Sequential
from torchvision import models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ClassificationNet(nn.Module):
    def __init__(self, image_size, n_classes, name="resnet34"):
        super(ClassificationNet, self).__init__()

        self.pre_net = getattr(models, name)(pretrained=True)
        if name == "alexnet":
            self.pre_net.classifier[6] = Identity()
            self.classifier1 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(4096, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, n_classes[0]),
            )
            self.classifier2 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(4096, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, n_classes[1]),
            )
            self.classifier3 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(4096, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, n_classes[2]),
            )
        elif name.startswith("resnet"):
            self.pre_net.fc = Identity()
            self.classifier1 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, n_classes[0]),
            )
            self.classifier2 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, n_classes[1]),
            )
            self.classifier3 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, n_classes[2]),
            )

        for child in self.pre_net.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, input):
        w = self.pre_net(input)
        x = self.classifier1(w)
        y = self.classifier2(w)
        z = self.classifier3(w)
        return (x, y, z)

    def oneshot(model, device, data):
        model.eval()

        with torch.no_grad():
            for i in range(len(data)):
                data[i] = data[i].to(device)

            output = model(data)
            return torch.squeeze(torch.argmax(output, dim=1)).cpu().item()

    def set_as_feature_extractor(self):
        self.pre_net.classifier = Sequential()


class SiameseNet(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.pre_net = alexnet(pretrained=True)
        for child in self.pre_net.children():
            for param in child.parameters():
                param.requires_grad = False

        self.pre_net.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 32),
            nn.ReLU(inplace=True),
        )
        for param in self.pre_net.classifier.parameters():
            param.requires_grad = True

        self.classifier = nn.Linear(32, 1)

    def forward(self, data):
        res = []
        for i in range(2):  # Siamese nets; sharing weights
            x = data[:, i]
            x = self.pre_net(x)
            res.append(x)

        res = torch.abs(res[1] - res[0])
        res = self.classifier(res)
        return res

    def oneshot(model, device, data):
        model.eval()

        with torch.no_grad():
            for i in range(len(data)):
                data[i] = data[i].to(device)

            output = model(data)
            return torch.squeeze(torch.argmax(output, dim=1)).cpu().item()


def set_as_feature_extractor(model):
    model.pre_net.classifier = Sequential()
