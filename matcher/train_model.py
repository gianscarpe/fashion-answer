import torch
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from matcher.models import Net
from matcher.dataset import SiameseDataset
from torch.utils.data import DataLoader
import time


def main():
    config = {
        'save_frequency': 2,
        'batch_size': 16,
        'lr': 0.001,
        'num_epochs': 10,
        'weight_decay': 0.0001,
        'exp_base_dir': 'data/exps',
        'image_size': [60, 80]
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    model = Net(image_size=config['image_size']).to(device)

    train_loader = DataLoader(
        SiameseDataset('data/fashion-product-images-small/images',
                       'data/fashion-product-images-small/small_train.csv',
                       load_path='data/fashion-product-images-small/train',
                       image_size=config['image_size']),
        batch_size=config['batch_size'], shuffle=True)

    val_loader = DataLoader(
        SiameseDataset('data/fashion-product-images-small/images',
                       'data/fashion-product-images-small/small_val.csv',
                       load_path='data/fashion-product-images-small/val',
                       image_size=config['image_size']),
        batch_size=config['batch_size'], shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    for epoch in range(config['num_epochs']):
        train(model, device, train_loader, epoch, optimizer, config['batch_size'])
        test(model, device, val_loader)
        if epoch & config['save_frequency'] == 0:
            torch.save(model, 'siamese_{:03}.pt'.format(epoch))


def train(model, device, train_loader, epoch, optimizer, batch_size):
    model.train()
    t0 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        for i in range(len(data)):
            data[i] = data[i].to(device)

        optimizer.zero_grad()
        output_positive = model(data[:, 0])
        output_negative = model(data[:, 1])

        target = target.type(torch.LongTensor).to(device)
        target_positive = torch.squeeze(target[:, 0])
        target_negative = torch.squeeze(target[:, 1])

        loss_positive = F.cross_entropy(output_positive, target_positive)
        loss_negative = F.cross_entropy(output_negative, target_negative)

        loss = loss_positive + loss_negative
        loss.backward()

        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} {:.1f}s [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (time.time() - t0), batch_idx * batch_size, len(train_loader.dataset),
                                           100. * batch_idx * batch_size / len(
                                               train_loader.dataset), loss.item()))


def test(model, device, test_loader):
    model.eval()

    with torch.no_grad():
        accurate_labels = 0
        all_labels = 0
        loss = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to(device)

            output_positive = model(data[:, 0])
            output_negative = model(data[:, 1])

            target = target.type(torch.LongTensor).to(device)
            target_positive = torch.squeeze(target[:, 0])
            target_negative = torch.squeeze(target[:, 1])

            loss_positive = F.cross_entropy(output_positive, target_positive)
            loss_negative = F.cross_entropy(output_negative, target_negative)

            loss = loss + loss_positive + loss_negative

            accurate_labels_positive = torch.sum(
                torch.argmax(output_positive, dim=1) == target_positive).cpu()
            accurate_labels_negative = torch.sum(
                torch.argmax(output_negative, dim=1) == target_negative).cpu()

            accurate_labels = accurate_labels + accurate_labels_positive + accurate_labels_negative
            all_labels = all_labels + len(target_positive) + len(target_negative)

        accuracy = 100. * accurate_labels / all_labels
        print('Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(accurate_labels, all_labels,
                                                                    accuracy, loss))


if __name__ == '__main__':
    main()
