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
        self, image_size, n_classes_phase1, n_classes_phase2, phase="1", name="resnet34"
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

    def phase2(self, model_path):
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
    def __init__(self, image_size, n_classes, name="resnet34", method="classification"):
        super(ClassificationNet, self).__init__()

        self.name = name
        self.method = method
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

    def set_as_feature_extractor(self, name="alexnet"):
        try:
            model_name = self.name
        except AttributeError:
            model_name = name
        self.method = "features"
        self.classifier1[-1] = Identity()
        self.classifier2[-1] = Identity()
        self.classifier3[-1] = Identity()
