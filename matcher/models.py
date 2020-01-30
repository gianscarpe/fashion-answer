import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        output_size = [(image_size[0] - 2) / 2, (image_size[1] - 2) / 2]
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3)

        self.pool1 = nn.MaxPool2d(2)

        self.linear1 = nn.Linear(640, 16)

        self.linear2 = nn.Linear(16, 2)

    def forward(self, data):
        res = []
        for i in range(2):  # Siamese nets; sharing weights
            x = data[:, i]
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool1(x)
            x = self.conv3(x)
            x = F.relu(x)
            x = self.pool1(x)

            x = x.view(x.shape[0], -1)
            x = self.linear1(x)
            res.append(F.relu(x))

        res = torch.abs(res[1] - res[0])
        res = self.linear2(res)
        return res


def oneshot(model, device, data):
    model.eval()

    with torch.no_grad():
        for i in range(len(data)):
            data[i] = data[i].to(device)

        output = model(data)
        return torch.squeeze(torch.argmax(output, dim=1)).cpu().item()
