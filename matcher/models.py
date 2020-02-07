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
        if self.method == "classification":
            x = self.classifier1(w)
            y = self.classifier2(w)
            z = self.classifier3(w)
            return (x, y, z)
        elif self.method == "features":
            return w

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
        if model_name == "alexnet":
            self.pre_net.classifier = Identity()
        elif model_name.startswith("resnet"):
            self.pre_net.fc = Identity()
