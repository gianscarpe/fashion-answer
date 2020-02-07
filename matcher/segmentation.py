# -*- coding: utf-8 -*-

import os
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torchvision.transforms.functional as TF
import torch
import numpy as np
import segmentation_models_pytorch as smp

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """

    CLASSES = ["dress", "not dress"]

    def __init__(
        self,
        images_dir,
        masks_dir,
        classes=None,
        augmentation=None,
        preprocessing=None,
        im_resize=None,
    ):
        self.ids = os.listdir(images_dir)
        self.mask_ids = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.im_resize = im_resize

    def __getitem__(self, i):

        # read data

        image = cv2.resize(
            cv2.cvtColor(cv2.imread(self.images_fps[i]), cv2.COLOR_BGR2RGB),
            self.im_resize,
        )

        mask = np.load(self.masks_fps[i])
        mask = cv2.resize((mask > 0).astype("double"), self.im_resize)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            image = self.preprocessing(image)

        image = TF.to_tensor(image).type(torch.FloatTensor)
        mask = TF.to_tensor(mask).type(torch.FloatTensor)

        return image, mask

    def __len__(self):
        return len(self.ids)


"""## Create model and train"""

ENCODER = "mobilenet_v2"
ENCODER_WEIGHTS = "imagenet"
CLASSES = ["dress"]
ACTIVATION = (
    "sigmoid"
)  # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = "cpu"
PHOTO_DIR = "data/fashion-product-images-small/segmentation/photos"
ANNOTATION_DIR = "data/fashion-product-images-small/segmentation/numpy"

x_train_dir = os.path.join(PHOTO_DIR)
y_train_dir = os.path.join(ANNOTATION_DIR)
x_valid_dir = os.path.join(PHOTO_DIR)
y_valid_dir = os.path.join(ANNOTATION_DIR)

if __name__ == "__main__":
    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    # model.encoder.set_swish(memory_efficient=True)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=None,
        preprocessing=preprocessing_fn,
        classes=CLASSES,
        im_resize=(768, 512),
    )

    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=None,
        preprocessing=None,
        classes=CLASSES,
        im_resize=(768, 512),
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    loss = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.5)]

    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    # train model for 40 epochs

    max_score = 0

    for i in range(0, 40):

        print("\nEpoch: {}".format(i))
        train_logs = train_epoch.run(train_loader)

        # do something (save model, change lr, etc.)
        if max_score < train_logs["iou_score"]:
            max_score = train_logs["iou_score"]
            torch.save(model, "./best_model.pth")
            print("Model saved!")

        if i == 25:
            optimizer.param_groups[0]["lr"] = 1e-5
            print("Decrease decoder learning rate to 1e-5!")
