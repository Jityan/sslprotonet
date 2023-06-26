import os
import pickle
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

ROOT_PATH = './data/cifar-fs'

def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data


class CIFAR_FS(Dataset):

    def __init__(self, phase='train', size=32, transform=None):

        filepath = os.path.join(ROOT_PATH, 'CIFAR_FS_' + phase + ".pickle")
        datafile = load_data(filepath)
        
        data = datafile['data']
        label = datafile['labels']

        data = [Image.fromarray(x) for x in data]

        min_label = min(label)
        label = [x - min_label for x in label]

        newlabel = []
        classlabel = 0
        for i in range(len(label)):
            if (i > 0) and (label[i] != label[i-1]):
                classlabel += 1
            newlabel.append(classlabel)

        self.data = data
        self.label = newlabel

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                    np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
                    )
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]


class SSLCifarFS(Dataset):

    def __init__(self, phase, args):
        filepath = os.path.join(ROOT_PATH, 'CIFAR_FS_' + phase + ".pickle")
        datafile = load_data(filepath)
        
        data = datafile['data']
        label = datafile['labels']

        data = [Image.fromarray(x) for x in data]

        min_label = min(label)
        label = [x - min_label for x in label]

        newlabel = []
        classlabel = 0
        for i in range(len(label)):
            if (i > 0) and (label[i] != label[i-1]):
                classlabel += 1
            newlabel.append(classlabel)

        self.data = data
        self.label = newlabel
        self.args = args

        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                              saturation=0.4, hue=0.1)
        self.augmentation_transform = transforms.Compose([transforms.RandomResizedCrop(size=(args.size, args.size)[-2:],
                        scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
                )
        ])
        #
        self.identity_transform = transforms.Compose([
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
                )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, label = self.data[i], self.label[i]
        image = []
        for _ in range(self.args.shot):
            image.append(self.identity_transform(img).unsqueeze(0))
        for i in range(self.args.train_query):
            image.append(self.augmentation_transform(img).unsqueeze(0))
        return dict(data=torch.cat(image)), label

