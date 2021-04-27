""" Module for the data loading pipeline for the model to train """
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import cv2
import numpy as np
import os
import random


class imgdataset(VisionDataset):
    def __init__(self, rootlist, process=None, transform=None, randomdrop=0):
        super(imgdataset, self).__init__(root="", transform=transform)
        self.rootlist = rootlist
        self.randomdrop = randomdrop
        self.dataset = []
        self.process = process
        for root, label in self.rootlist:
            imglist = os.listdir(root)
            print("Loading %s" % (root), end="\r")
            for p in imglist:
                self.dataset.append((os.path.join(root, p), label))
            print("Loaded %s=>%d" % (root, len(imglist)))

    def shuffle(self):
        random.shuffle(self.dataset)

    def reset(self):
        self.dataset = []
        for root, label in self.rootlist:
            imglist = os.listdir(root)
            for p in imglist:
                self.dataset.append((os.path.join(root, p), label))

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = Image.open(img)
        img = np.array(img)
        img = cv2.resize(img, (256, 256))
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def __add__(self, other):
        self.dataset.extend(other.dataset)
        return self
