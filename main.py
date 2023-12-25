import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import argparse
import tqdm
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image
from datetime import datetime


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--exp_dir', type=str, default='logs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs_uns', type=int, default=2)
    parser.add_argument('--epochs_sup', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr_uns', type=float, default=0.06)
    parser.add_argument('--lr_sup', type=float, default=30)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--save_exp', type=bool, default=True)
    return parser.parse_args()


def get_path(args):
    if args.save_exp:
        path = args.exp_dir
        if not os.path.exists(path):
            os.mkdir(path)
        path += datetime.now().strftime('%Y-%m-%d_%H-%M-%S/')
        if not os.path.exists(path):
            os.mkdir(path)
    else:
        path = '/'
    args.path = path


def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_data(args):
    class SiameseTransform:
        def __init__(self, transform):
            self.transform = transform

        def __call__(self, img):
            return self.transform(img), self.transform(img)

    transform_train_uns = SiameseTransform(transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
    transform_train_sup = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(int(32 * (8 / 7)), interpolation=Image.BICUBIC),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train_uns = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform_train_uns)
    dataset_train_sup = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform_train_sup)
    dataset_train = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform_test)
    dataset_test = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform_test)
    loader_train_uns = torch.utils.data.DataLoader(dataset_train_uns, batch_size=args.batch_size, num_workers=5,
                                                   shuffle=True, pin_memory=True, drop_last=True)
    loader_train_sup = torch.utils.data.DataLoader(dataset_train_sup, batch_size=args.batch_size, num_workers=5,
                                                   shuffle=True, pin_memory=True, drop_last=True)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=5,
                                               shuffle=False, pin_memory=True)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=5,
                                              shuffle=False, pin_memory=True)
    return loader_train_uns, loader_train_sup, loader_train, loader_test


def get_backbone():
    backbone = models.resnet18()
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, dim


def check_parameters(*args):
    for model in args:
        parameters = sum(param.numel() for param in model.parameters())
        # print(model)
        print('Parameters:', parameters / 10 ** 6, 'MB')


def train_uns(model, loader, optimizer, scheduler, args):
    model.train()
    losses = []
    lrs = []
    global_progress = tqdm.tqdm(range(args.epochs_uns), desc='Training_uns')
    for epoch in global_progress:
        loss_ = 0
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        local_progress = tqdm.tqdm(loader, desc=f'Epoch {epoch}/{args.epochs_uns}')
        for (img1, img2), _ in local_progress:
            model.zero_grad()
            loss = model(img1.to(args.device), img2.to(args.device))
            loss.backward()
            optimizer.step()
            local_progress.set_postfix({"loss": loss.item(), "lr": lr})
            loss_ += loss.item() * len(img1)
        losses.append(loss_ / len(loader.dataset))
        lrs.append(lr)
        scheduler.step()

    x = range(1, args.epochs_uns + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plot1 = ax1.plot(x, losses, '-', label="Loss")
    ax2 = ax1.twinx()
    plot2 = ax2.plot(x, lrs, '-r', label="Learning Rate")
    plots = plot1 + plot2
    labs = [p.get_label() for p in plots]
    ax1.legend(plots, labs)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Loss per Epoch")
    ax1.set_ylim(-1, 0)
    ax1.set_yticks([i / 10. for i in range(-10, 1)])
    ax1.grid(axis='y')
    # ax2.set_ylabel("Learning Rate")
    ax2.set_ylim(-0.1 * args.lr_uns, 1.1 * args.lr_uns)
    plt.title("Training_uns")
    plt.savefig(f"{args.path}Training_uns_{args.epochs_uns}.png")


def train_sup(backbone, classifier, loader, optimizer, scheduler, args, loader_train=None, loader_test=None):
    backbone.eval()
    classifier.train()
    losses = []
    accuracies = []
    accuracies_train = []
    accuracies_test = []
    lrs = []
    total = len(loader.dataset)
    global_progress = tqdm.tqdm(range(args.epochs_sup), desc='Training_sup')
    for epoch in global_progress:
        loss_ = 0
        correct = 0
        accuracy = 0
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        local_progress = tqdm.tqdm(loader, desc=f'Epoch {epoch}/{args.epochs_sup}')
        for idx, (data, target) in enumerate(local_progress):
            classifier.zero_grad()
            with torch.no_grad():
                feature = backbone(data.to(args.device))
            output = classifier(feature)
            loss = F.cross_entropy(output, target.to(args.device))
            loss.backward()
            optimizer.step()
            pred = F.log_softmax(output, dim=1).argmax(dim=1)
            correct += (pred == target.to(args.device)).sum().item()
            accuracy = 100. * correct / total
            loss_ += loss.item() * len(target)
            local_progress.set_postfix({"loss": loss.item(), "lr": lr, "Accuracy": "{:.2f}%".format(accuracy)})
            accuracy_train = 0
            if idx == len(loader) - 1 and loader_train:
                correct_train = 0
                with torch.no_grad():
                    for data_train, target_train in loader_train:
                        pred = F.log_softmax(classifier(backbone(data_train.to(args.device))), dim=1).argmax(dim=1)
                        correct_train += (pred == target_train.to(args.device)).sum().item()
                        accuracy_train = 100. * correct_train / len(loader_train.dataset)
                        local_progress.set_postfix({"loss": loss_ / total, "lr": lr,
                                                    "Accuracy": "{:.2f}%/{:.2f}%".format(accuracy, accuracy_train)})
                accuracies_train.append(accuracy_train)
            accuracy_test = 0
            if idx == len(loader) - 1 and loader_test:
                correct_test = 0
                with torch.no_grad():
                    for data_test, target_test in loader_test:
                        pred = F.log_softmax(classifier(backbone(data_test.to(args.device))), dim=1).argmax(dim=1)
                        correct_test += (pred == target_test.to(args.device)).sum().item()
                        accuracy_test = 100. * correct_test / len(loader_test.dataset)
                        local_progress.set_postfix({"loss": loss_ / total, "lr": lr,
                                                    "Accuracy": "{:.2f}%/{:.2f}%/{:.2f}%".format(accuracy,
                                                                                                 accuracy_train,
                                                                                                 accuracy_test)})
                accuracies_test.append(accuracy_test)
        losses.append(loss_ / total)
        accuracies.append(accuracy)
        lrs.append(lr)
        scheduler.step()

    x = range(1, args.epochs_sup + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plot1 = ax1.plot(x, accuracies, '-', label="Train Accuracy")
    plots = plot1
    if loader_train:
        plot2 = ax1.plot(x, accuracies_train, '-', label="Test Accuracy of Train")
        plots = plots + plot2
    if loader_test:
        plot3 = ax1.plot(x, accuracies_test, '-', label="Test Accuracy of Test")
        plots = plots + plot3
    ax2 = ax1.twinx()
    plot4 = ax2.plot(x, lrs, '-r', label="Learning Rate")
    plots = plots + plot4
    labs = [p.get_label() for p in plots]
    ax1.legend(plots, labs)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy(%)")
    ax1.set_ylim(85, 95)
    ax1.set_yticks(range(85, 96))
    ax1.grid(axis='y')
    # ax2.set_ylabel("Learning Rate")
    ax2.set_ylim(-0.1 * args.lr_sup, 1.1 * args.lr_sup)
    plt.title("Training_sup")
    plt.savefig(f"{args.path}Training_sup_{args.epochs_sup}.png")


def test(backbone, classifier, loader_train, loader_test, args):
    backbone.eval()
    classifier.eval()
    correct_train = 0.
    correct_test = 0.
    result = torch.zeros(10, 10)
    progress_train = tqdm.tqdm(loader_train, desc='Testing_train')
    with torch.no_grad():
        for data, target in progress_train:
            pred = F.log_softmax(classifier(backbone(data.to(args.device))), dim=1).argmax(dim=1)
            correct_train += (pred == target.to(args.device)).sum().item()
            progress_train.set_postfix({"Accuracy": "{:.2f}%".format(100. * correct_train / len(loader_train.dataset))})
    progress_test = tqdm.tqdm(loader_test, desc='Testing_test')
    with torch.no_grad():
        for data, target in progress_test:
            pred = F.log_softmax(classifier(backbone(data.to(args.device))), dim=1).argmax(dim=1)
            for i in range(len(target)):
                result[target[i]][pred[i]] += 1
            correct_test += (pred == target.to(args.device)).sum().item()
            progress_test.set_postfix({"Accuracy": "{:.2f}%".format(100. * correct_test / len(loader_test.dataset))})

    labels = ['Plane', 'Car ', 'Bird', 'Cat ', 'Deer', 'Dog ', 'Frog', 'Horse', 'Ship', 'Truck']
    print('Tgt\\Prd', *labels, 'RECALL(%)', sep='\t')
    for i, t in enumerate(result / 1000):
        print(labels[i], end='\t')
        for j, p in enumerate(t):
            print('{:.3f}'.format(p), '*' if i == j else ' ', end='\t', sep='')
        print('{:.1f}%'.format(t[i] * 100))
    print('SUM ', end='\t')
    for i in result.sum(dim=0):
        print('{:.3f}'.format(i / 1000), end='\t')
    print('\nPRE(%)', end='\t')
    for i in range(10):
        r = 0.1 * result[i][i]
        p = 1000 * r / result.sum(dim=0)[i] if result.sum(dim=0)[i] else 100
        print('{:.2f}%'.format(p), end='\t')
    print()
    if args.save_exp:
        with open('{}result_{:.2f}%_{}%.txt'.format(args.path, correct_train / 500, correct_test / 100), 'w') as f:
            print('Tgt\\Prd', *labels, 'RECALL(%)', sep='\t', file=f)
            for i, t in enumerate(result / 1000):
                print(labels[i], end='\t', file=f)
                for j, p in enumerate(t):
                    print('{:.3f}'.format(p), '*' if i == j else ' ', end='\t', sep='', file=f)
                print('{:.1f}%'.format(t[i] * 100), file=f)
            print('SUM ', end='\t', file=f)
            for i in result.sum(dim=0):
                print('{:.3f}'.format(i / 1000), end='\t', file=f)
            print('\nPRE(%)', end='\t', file=f)
            for i in range(10):
                r = 0.1 * result[i][i]
                p = 1000 * r / result.sum(dim=0)[i] if result.sum(dim=0)[i] else 100
                print('{:.2f}%'.format(p), end='\t', file=f)


def main():
    args = get_args()
    get_path(args)
    set_seeds(args.seed)
    loader_train_uns, loader_train_sup, loader_train, loader_test = get_data(args)
    backbone, dim = get_backbone()
    model = SimSiam(backbone, dim).to(args.device)
    classifier = nn.Linear(in_features=dim, out_features=10, bias=True).to(args.device)
    # check_parameters(model, backbone, model.projector, model.predictor, classifier)
    optimizer_uns = torch.optim.SGD(model.parameters(), lr=args.lr_uns, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    scheduler_uns = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_uns, T_max=args.epochs_uns, eta_min=0)
    optimizer_sup = torch.optim.SGD(classifier.parameters(), lr=args.lr_sup, momentum=args.momentum)
    scheduler_sup = torch.optim.lr_scheduler.StepLR(optimizer_sup, step_size=40, gamma=0.6)
    # optimizer_uns = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    # optimizer_sup = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)
    flag_uns, flag_sup = True, True
    if flag_uns:
        set_seeds(args.seed)
        train_uns(model, loader_train_uns, optimizer_uns, scheduler_uns, args)
        torch.save(model.state_dict(), f"{args.path}model.pt")
    else:
        model.load_state_dict(torch.load(f"model.pt"))
    if flag_sup:
        set_seeds(args.seed)
        train_sup(backbone, classifier, loader_train_sup, optimizer_sup, scheduler_sup, args, loader_train=loader_train,
                  loader_test=loader_test)
        torch.save(classifier.state_dict(), f"{args.path}classifier.pt")
    else:
        classifier.load_state_dict(torch.load(f"classifier.pt"))
    test(backbone, classifier, loader_train, loader_test, args)
    if args.save_exp:
        shutil.copyfile(__file__, f'{args.path}main.py')


class ProjectionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class PredictionMLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimSiam(nn.Module):
    def __init__(self, backbone, input_dim):
        super().__init__()
        self.backbone = backbone
        self.projector = ProjectionMLP(input_dim=input_dim)
        self.predictor = PredictionMLP()

    def forward(self, x1, x2):
        x0 = (x1 + x2) / 2
        z0 = self.projector(self.backbone(x0))
        z1, z2 = self.projector(self.backbone(x1)), self.projector(self.backbone(x2))
        z = torch.maximum(z1, z2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        p0 = self.predictor(z0)
        # loss = self.d(p1, z2) / 2 + self.d(p2, z1) / 2
        loss = self.d(p1, z2) / 4 + self.d(p2, z1) / 4 + self.d(p0, z) / 2
        return loss

    def d(self, p, z):
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()


if __name__ == '__main__':
    main()
