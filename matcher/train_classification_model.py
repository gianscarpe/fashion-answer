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
        'save_frequency': 2,
        'batch_size': 16,
        'lr': 0.001,
        'num_epochs': 10,
        'weight_decay': 0.0001,
        'exp_base_dir': 'data/exps/exp1',
        'image_size': [224, 224],
        'load_path': None,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_epoch = 1

    train_loader = DataLoader(
        ClassificationDataset('./data/images/',
                              './data/small_train.csv',
                              load_path=None,
                              image_size=config['image_size']),
        batch_size=config['batch_size'], shuffle=True)

    val_loader = DataLoader(
        ClassificationDataset('./data/images',
                              './data/small_val.csv',
                              image_size=config['image_size'],
                              thr=5),
        batch_size=config['batch_size'], shuffle=True)

    model = ClassificationNet(image_size=config['image_size'],
                              n_classes=train_loader.dataset.n_classes).to(device)

    if config['load_path']:
        print("Loading and evaluating model")
        start_epoch = int(config['load_path'][-6:-3])
        model = torch.load(config['load_path'])
        test(model, device, val_loader)

    optimizer = optim.Adam(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    for epoch in range(start_epoch, config['num_epochs']):
        train(model, device, train_loader, epoch,
              optimizer, config['batch_size'])
        test(model, device, val_loader)
        if epoch % config['save_frequency'] == 0:
            torch.save(model, os.path.join(config['exp_base_dir'],
                                           'classification_{:03}.pt'.format(epoch)))


def train(model, device, train_loader, epoch, optimizer, batch_size):
    model.train()
    t0 = time.time()
    training_loss = []
    criterion = CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        for i in range(len(data)):
            data[i] = data[i].to(device)

        optimizer.zero_grad()
        output = torch.squeeze(model(data))

        loss = criterion(output, target)
        training_loss.append(loss.item())
        loss.backward()

        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \tBatch Loss: {:.6f}'.format(
                epoch + 1, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx * batch_size / len(
                    train_loader.dataset), loss))
    print('Train Epoch: {}\t time:{:.3f}s \tMeanLoss: {:.6f}'.format(
        epoch + 1, (time.time() - t0), np.average(training_loss)))


def test(model, device, test_loader):
    model.eval()

    with torch.no_grad():
        accurate_labels = 0
        all_labels = 0
        val_loss = []
        for batch_idx, (data, target) in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to(device)

            output = torch.squeeze(model(data))

            val_loss.append(F.cross_entropy(output, target))

            accurate_labels += torch.sum(
                (torch.argmax(F.softmax(output), dim=1) == target))
            all_labels += len(target)

        accuracy = 100. * accurate_labels.cpu().numpy() / all_labels
        print('Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(accurate_labels, all_labels,
                                                                    accuracy,
                                                                    np.average(val_loss)))


if __name__ == '__main__':
    main()
