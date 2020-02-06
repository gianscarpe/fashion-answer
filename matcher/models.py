import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential
from torchvision.models import alexnet

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import alexnet


class ClassificationNet(nn.Module):
    def __init__(self, image_size, n_classes):
        super().__init__()

        self.pre_net = alexnet(pretrained=True)

        for child in self.pre_net.children():
            for param in child.parameters():
                param.requires_grad = False

        self.classifier1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, n_classes[0]),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, n_classes[1]),
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, n_classes[2]),
        )

        for param in self.pre_net.classifier.parameters():
            param.requires_grad = True

    def forward(self, data):
        x = data
        x = self.pre_net.features(x)
        x = self.pre_net.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x)
        w = self.classifier1(x)
        y = self.classifier2(x)
        z = self.classifier3(x)
        return (w, y, z)

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
