import random, json, csv, os, cv2, io
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
import torchaudio
import librosa
from PIL import Image
from . import video_transforms as vtransforms
from scipy.io import wavfile as wf
from torchvision.transforms import InterpolationMode


class BaseDataset(torchdata.Dataset):
    def __init__(self, data_list_path, args, mode):
        self.imgSize = args.imgSize
        self.mode = mode
        self._init_vtransform()

        self.root = '/'.join(data_list_path.split('/')[:-1]) + '/'
        if mode == 'train' or mode == 'val':
            with open(data_list_path) as f:
                self.data_list = json.load(f)
        elif mode == 'test':
            self.data_list = [self.root + '/' + x for x in os.listdir(data_list_path)]

        with open(args.label_path) as f:
            self.id2cate = json.load(f)

        print('# %s samples: %d' % (mode, len(self.data_list)))

    def __getitem__(self, idx, ):
        path = self.data_list[idx]
        try:
            img = self._load_frame(path)
        except Exception as E:
            print(E)
            path = self.data_list[random.choice(range(len(self.data_list)))]
            img = self._load_frame(path)

        ID = path.split('/')[-1]

        ret = {'img': img}
        if self.mode != 'test':
            label = self.id2cate[ID]
            ret['label'] = label
        else:
            ret['ID'] = ID

        return ret

    def __len__(self):
        return len(self.data_list)

    def _init_vtransform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.mode == 'train':
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.Resize((int(self.imgSize * 1.1), int(self.imgSize * 1.1)), InterpolationMode.BICUBIC))
            # transform_list.append(transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC))
            transform_list.append(transforms.RandomCrop(self.imgSize))
        else:
            transform_list.append(transforms.Resize((self.imgSize, self.imgSize), InterpolationMode.BICUBIC))
            # transform_list.append(transforms.Resize((self.imgSize), Image.BICUBIC))
            # transform_list.append(transforms.CenterCrop(self.imgSize))

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean, std))
        self.img_transform = transforms.Compose(transform_list)

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        # img = cv2.imread(path)
        img = self.img_transform(img)
        return img
