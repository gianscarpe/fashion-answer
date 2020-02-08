from PIL import Image
from matcher.dataset import SiameseDataset
from torchvision import transforms
from matplotlib import pyplot as plt
import torch

dataset = SiameseDataset(
    "data/fashion-product-images-small/images",
    "data/fashion-product-images-small/small_train.csv",
    distinguish_class="masterCategory",
    image_size=[40, 80],
)
dataset.save("data/fashion-product-images-small/train")

for ind in range(0, 10):
    images, target = dataset[ind]
    positive = images[1]
    img1 = torch.squeeze(positive[0])
    img2 = torch.squeeze(positive[1])

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(img1)  # row=0, col=0
    ax[1].imshow(img2)
    plt.show()
