import torch
from torch import nn
from torchvision import models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class TwoPhaseNet(nn.Module):
    def __init__(
        self, image_size, n_classes_phase1, n_classes_phase2, phase="1", name="resnet18"
    ):
        super(TwoPhaseNet, self).__init__()

        self.image_size = image_size
        self.n_classes_phase1 = n_classes_phase1
        self.n_classes_phase2 = n_classes_phase2
        self.phase = phase
        resnet_num = name.split("resnet")[1]
        if resnet_num != "18" and resnet_num != "34":
            raise Exception("Only resnet18 and resnet34 are supported")
        self.name = name
        self.pre_net = getattr(models, name)(pretrained=True)
        self.pre_net.fc = Identity()
        self.features = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
        )
        self.classifier = Identity()

    def phase1(self):
        for child in self.pre_net.children():
            for param in child.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.n_classes_phase1),
        )

    def phase2(self):
        for child in self.pre_net.children():
            for param in child.parameters():
                param.requires_grad = True
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.n_classes_phase2),
        )

    def forward(self, x):
        x = self.pre_net(x)
        x = self.features(x)
        x = self.classifier(x)
        return x


class ClassificationNet(nn.Module):
    def __init__(self, image_size, n_classes: list, name="resnet34"):
        super(ClassificationNet, self).__init__()

        self.name = name
        self.n_classes = n_classes
        if name == "alexnet":
            self.pre_net = getattr(models, name)(pretrained=True)
            self.features = nn.Sequential(
                nn.Linear(4096, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
            )
        elif name.startswith("resnet"):
            self.pre_net = getattr(models, name)(pretrained=True)
            self.pre_net.fc = Identity()
            self.features = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
            )
        for child in self.pre_net.children():
            for param in child.parameters():
                param.requires_grad = False
        self.classifier()

    def extract_features(self):
        for i in range(len(self.n_classes)):
            setattr(self, "classifier" + str(i), Identity())

    def classifier(self):
        for i in range(len(self.n_classes)):
            setattr(
                self,
                "classifier" + str(i),
                nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, self.n_classes[i]),
                ),
            )

    def forward(self, x):
        x = self.pre_net(x)
        feat = self.features(x)
        results = []
        for i in range(len(self.n_classes)):
            r = getattr(self, "classifier" + str(i))(feat)
            results.append(r)
        return tuple(results)
