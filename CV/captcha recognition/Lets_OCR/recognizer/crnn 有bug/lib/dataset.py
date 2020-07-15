#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
from torchvision import transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import base64


class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:

            str = 'num-samples'
            nSamples = int(txn.get(str.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            imgbuf = base64.b64decode(imgbuf)  # add 20200626
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                # img = Image.open(buf) .convert('L') # origin version

                # 20200630_add begin
                img = Image.open(buf)
                # print('img.size[0]',img.size[0]) # 200
                # print('img.size[1]',img.size[1]) # 100
                img = img.resize((100, 32), Image.ANTIALIAS)
                width = img.size[0]
                height = img.size[1]
                array = []
                for x in range(width):
                    for y in range(height):
                        r, g, b = img.getpixel((x, y))
                        rgb = (r, g, b)
                        array.append(rgb)
                img = np.array(array).reshape((width, height,3)) # W H C
                img = np.fliplr(img)
                img = np.rot90(img)
                # print('==========')
                # print(img[:,:,1][0])
                # img = Image.fromarray(img, dtype=np.uint8)
                # 20200630_add end

            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)


            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode())

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # img = self.toTensor(img)
        # img = img.permute(2, 0, 1)
        # #img = torch.from_numpy(img.permute(2, 0, 1))
        # # img = img.resize(self.size, self.interpolation)
        # img = img.float().div(255).sub_(0.5).div_(0.5)
        # img = img.transpose((2, 0, 1))
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        # print(img)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=256, keep_ratio=False, min_ratio=1):

        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.toTensor = transforms.ToTensor() # 20200701 add

    def __call__(self, batch):
        #print("batch = ", batch)

        images, labels = zip(*batch)
        # print("####l129 ,images=", images)
        # print("####l130 ,len_images=", len(images))
        # print("####l131 ,len_labels=", len(labels))
        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))#20200626
        images = [transform(image) for image in images]
        images = torch.cat([self.toTensor(t).unsqueeze(0) for t in images], 0)
        #images = torch.cat([self.toTensor(t) for t in images], 0)
        # imgs_1 = []
        # for image in images:
        #     img_1 = transform(image)
        #     imgs_1.append(img_1)
        #
        # imgs_2 = []
        # for t in imgs_1:
        #     img_2 = t.unsqueeze(0)
        #     imgs_2.append(img_2)

        # images = torch.cat(imgs_2, 0)
        # print("===============",images.shape)
        return images, labels


def loadData(v, data):
    # v.data.resize_(data.size()).copy_(data)
    # print("V's shape:",v.shape)
    # print("V's value:", v)
    v.resize_(data.size()).copy_(data)  # 20200627 add