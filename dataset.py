import torch
import random
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler

def get_dataloaders(input_img_size, num_classes, num_tasks, classes_per_task, batch_size, shuffle, device):
    scale = (0.05, 1.0)
    ratio = (3./4., 4./3.)
    size = int((256/224) * input_img_size)
    pin = True if device == 'cuda' else False

    transforms_train =  transforms.Compose([
        transforms.RandomResizedCrop(input_img_size, scale=scale, ratio=ratio),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    transforms_test = transforms.Compose([
        transforms.Resize(size, interpolation=3),
        transforms.CenterCrop(input_img_size),
        transforms.ToTensor(),
    ])

    dataset_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms_train)
    dataset_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms_test)

    split_datasets = list()
    class_mask = list()
    dataloaders_train = list()
    dataloaders_test = list()
    labels = [i for i in range(num_classes)]

    if shuffle:
        random.shuffle(labels)

    for i in range(num_tasks):
        scope = labels[:classes_per_task]
        labels = labels[classes_per_task:]

        train_split_indices = []
        test_split_indices = []

        for j in range(len(dataset_train.targets)):
            if int(dataset_train.targets[j]) in scope:
                train_split_indices.append(j)
        
        for j in range(len(dataset_test.targets)):
            if int(dataset_test.targets[j]) in scope:
                test_split_indices.append(j)

        train_subset = Subset(dataset_train, train_split_indices)
        test_subset = Subset(dataset_test, test_split_indices)

        split_datasets.append([train_subset, test_subset])
        class_mask.append(scope)

    for i in range(num_tasks):
        dataset_train_temp, dataset_test_temp = split_datasets[i]

        train_sampler = RandomSampler(dataset_train_temp)
        test_sampler = SequentialSampler(dataset_test_temp)

        dataloader_train_temp = DataLoader(dataset_train_temp, sampler=train_sampler, batch_size=batch_size, pin_memory=pin)
        dataloader_test_temp = DataLoader(dataset_test_temp, sampler=test_sampler, batch_size=batch_size, pin_memory=pin)

        dataloaders_train.append(dataloader_train_temp)
        dataloaders_test.append(dataloader_test_temp)

    return dataloaders_train, dataloaders_test, class_mask