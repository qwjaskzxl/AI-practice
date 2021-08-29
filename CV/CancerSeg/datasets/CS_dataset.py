import random, json, csv, os, cv2, io
import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image
from scipy.io import wavfile as wf
from torchvision.transforms import InterpolationMode


class BaseDataset(torchdata.Dataset):
    def __init__(self, data_list_path, args, mode):
        self.imgSize = args.imgSize
        self.mode = mode
        self._init_transform()

        self.root = data_list_path
        data = os.listdir(self.root)

        if mode == 'train' or mode == 'val':
            self.data_list = data[:len(data) * 9 // 10]
        elif mode == 'test':
            self.data_list = data[len(data) * 9 // 10:]

        # with open(args.label_path) as f:
        #     self.id2cate = json.load(f)

        print('# %s samples: %d' % (mode, len(self.data_list)))

    def __getitem__(self, idx):
        img_path = self.root + self.data_list[idx]
        y_path = img_path.replace('original', 'target')

        img = self._load_img(img_path)
        y = self._load_img(y_path)
        # cv2.imwrite('vis/y.png' , y)
        # exit()

        # mask = self._load_mask(mask_path)
        if self.mode == 'train':
            img, y = self.v_h_flip(img, y)
        # mask[mask > 0.5] = 1
        # mask[mask <= 0.5] = 0

        # ID = img_path.split('/')[-1]
        # label = self.id2cate[ID]

        ret = {'ID': self.data_list[idx],
               'img': img,
               # 'label': label,
               'y': y}

        return ret

    def __len__(self):
        return len(self.data_list)

    def v_h_flip(self, img, y):
        if random.random() < 0.5:  # 左右翻转
            img = F.hflip(img)
            y = F.hflip(y)
        # if random.random() < 0.5:
        #     img = F.vflip(img)
        #     mask = F.vflip(mask)
        return img, y

    def _init_transform(self):
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]

        self.img_transform = transforms.Compose([  # transforms.Resize((self.imgSize), Image.BICUBIC),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
        ])
        # self.mask_transform = transforms.Compose([transforms.Resize((self.imgSize), Image.BICUBIC),
        #                                           transforms.ToTensor(), ])

    def _load_img(self, path):
        # img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        # img = torch.from_numpy(img)/ 255 - 0.5
        img = Image.open(path)#.convert('RGB')
        img = self.img_transform(img) - 0.5
        return img

    # def _load_mask(self, path):
    #     img = Image.open(path)
    #     # img = cv2.imread(path)
    #     img = self.mask_transform(img)
    #     return img
