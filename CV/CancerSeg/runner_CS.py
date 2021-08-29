import os, json, logging, collections
from copy import deepcopy
import torch, numpy as np
from torch.utils.data import DataLoader

from datasets.CS_dataset import BaseDataset
from models_ import CS_model
from utils import AverageMeter, TimeMeter
from sklearn.metrics import classification_report, f1_score
from tensorboardX import SummaryWriter
from time import time
from math import ceil
import cv2
import matplotlib.pyplot as plt
from PIL import Image


class Runner:
    def __init__(self, args):
        self.num_updates = 0
        self.args = args
        self._build_loader()
        self._build_model()
        self._build_optimizer()
        self.metric_meter = collections.defaultdict(lambda: AverageMeter())
        self.time_meter = TimeMeter()

    def train(self):
        args = self.args
        if not os.path.exists(self.args.model_saved_path):
            os.makedirs(self.args.model_saved_path)
        self.tb_writer = SummaryWriter('./tensorboard')

        best_acc = 0
        for epoch in range(1, self.args.max_num_epochs + 1):
            # meters = self.eval(dec=args.dec, epoch=epoch)
            logging.info('Start Epoch {}'.format(epoch))
            if self.args.evaluate:
                self.visualize()

            self._train_one_epoch(epoch)
            # self.eval(epoch)
            # self.tb_writer.add_scalar(self.args.mn + '_ctc_loss', metric_meter['ctc_loss'].avg, epoch)

            path = os.path.join(args.model_saved_path, 'model-%s' % (args.mn))
            torch.save(self.model.state_dict(), path)
            logging.info('model saved to %s' % path)
            self.print_log(epoch, reset=True)
        logging.info('Done.')

    def criteria(self, batch, output):
        label = batch['label'].cpu().numpy()
        B = label.shape[0]
        pred = torch.argmax(output['prob'], 1).cpu().numpy()
        LA = (pred == label).sum() / B

        y_mask = batch['y_mask'].cpu().numpy()
        map = output['logits_map'].detach().cpu().numpy()
        th = 0.5
        map[map > th] = 1.
        map[map != 1] = 0.
        # map[:] = 0
        N = B * (self.args.imgSize ** 2)
        PA = (map == y_mask).sum() / N
        return LA, PA

    def _train_one_epoch(self, epoch):

        self.model.train()
        for bid, batch in enumerate(self.train_loader, 1):

            img = batch['img'].cuda(non_blocking=True)
            y = batch['y'].cuda(non_blocking=True)
            output = self.model(img, y, train=True, epoch=epoch)
            loss = output['loss']  # .mean(), output['ar_loss'].mean(), output['ctc_loss'].mean(), output['rnnt_loss'].mean()
            # LA, PA = self.criteria(batch, output)
            # self.metric_meter['tra_LA'].update(LA)
            # self.metric_meter['tra_PA'].update(PA)

            self.optimizer.zero_grad()
            self.optimizer.backward(loss)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

            self.optimizer.step()
            self.num_updates += 1
            curr_lr = self.lr_scheduler.step_update(self.num_updates)

            self.metric_meter['loss'].update(loss.item())

            self.time_meter.update()

            # self.eval(epoch)

            # self.tb_writer.add_histogram(self.args.mn + '_loss', loss.item(), (epoch - 1) * len(self.train_loader) + bid)
            # print(self.args.mn + '_loss', loss.item())
            if bid % self.args.display_n_batches == 0 or bid == self.train_loader.__len__():
                self.print_log(epoch, bid, curr_lr, reset=False)

        return deepcopy(self.metric_meter)

    def visualize(self):
        with torch.no_grad():
            for bid, batch in enumerate(self.test_loader, 1):
                img = batch['img'].cuda(non_blocking=True)
                y = batch['y'].cuda(non_blocking=True)
                output = self.model(img, y, train=False)

                IDs = batch['ID']
                for i, ID in enumerate(IDs):
                    ID = ID.split('.')[0]
                    img = output['feat_map'][i].permute(1, 2, 0).contiguous().cpu().numpy()
                    img = img + 0.5
                    img[img > 1] = 1
                    img[img < 0] = 0
                    # print(img)
                    # exit()
                    # img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    # cv2.imwrite('vis/%s_pred.png' % (ID), img * 255)
                    plt.imsave('vis/%s_pred.png' % (ID), img)

                    img = batch['y'][i].permute(1, 2, 0).contiguous().cpu().numpy()
                    img = (img + 0.5)
                    # print(img.shape)
                    # exit()
                    # print(img.max())
                    # exit()
                    # img[img>1]=1

                    # img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    # cv2.imwrite('vis/%s_y.jpg' % (ID), img * 255)
                    # print(img.shape)
                    plt.imsave('vis/%s_y.png' % (ID), img)
                exit()

    def eval(self, epoch=None):
        meters = collections.defaultdict(lambda: AverageMeter())
        self.model.eval()

        with torch.no_grad():
            for bid, batch in enumerate(self.val_loader, 1):
                for k in batch.keys():
                    batch[k] = batch[k].cuda(non_blocking=True)
                output = self.model(**batch, train=False)
                LA, PA = self.criteria(batch, output)
                self.metric_meter['val_LA'].update(LA)
                self.metric_meter['val_PA'].update(PA)
                # if self.args.evaluate:
                #     print('loss:%.4f\tcer:%.4f\twer:%.4f\tAL:%.4f' % (loss, meters['CER'].avg, meters['WER'].avg, meters['AL'].avg))

            # print('| ', end='')
            # for key, value in meters.items():
            #     try:
            #         self.tb_writer.add_scalar(self.args.mn + '_' + key, value.avg, epoch)
            #     except Exception as E:
            #         print(E)
            #     print('{}, {:.4f}'.format(key, value.avg), end=' | ')
            #     meters[key].reset()
            # print()
        return deepcopy(meters)

    def print_log(self, epoch, bid=0, curr_lr=0, reset=False):
        msg = 'Epoch {}, Batch {}, lr = {:.5f}, '.format(epoch, bid, curr_lr)
        for k, v in self.metric_meter.items():
            msg += '{} = {:.4f}, '.format(k, v.avg)
            if reset:
                v.reset()
        msg += '{:.3f} seconds/batch'.format(1.0 / self.time_meter.avg)
        logging.info(msg)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logging.info('model saved to %s' % path)

    def _build_loader(self):
        args = self.args
        train = BaseDataset(args.train_data, args, 'train')  # args.dataset, args.vocab_path, args.text_max_length, args.use_word
        val = BaseDataset(args.val_data, args, 'val')
        test = BaseDataset(args.test_data, args, 'test')
        self.train_loader = DataLoader(dataset=train, batch_size=self.args.batch_size, num_workers=10,
                                       shuffle=True, pin_memory=False)
        self.val_loader = DataLoader(dataset=val, batch_size=self.args.batch_size, num_workers=10,
                                     shuffle=False) if val else None
        self.test_loader = DataLoader(dataset=test, batch_size=self.args.batch_size, num_workers=10,
                                      shuffle=False, pin_memory=True) if test else None

    def _build_model(self):
        self.model = CS_model.Model(self.args)
        # print(self.model)
        device_ids = [i for i in range(len(self.args.g.split(',')))]  # 指定后就一个gpu
        # self.model = self.model.to(torch.device('cpu'))
        if self.args.device == 'gpu':
            self.model = self.model.to(torch.device('cuda:%d' % device_ids[0]))
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        if self.args.load_pretrained_model:
            if self.args.partial_load:
                old_state_dict = self.model.state_dict()  # now
                new_state_dict = torch.load(self.args.model_state_dict_path)  # load
                cnt = 0
                for k in new_state_dict.keys():
                    if k in old_state_dict.keys() and 'visual_front_encoder' in k:
                        # print(k)
                        cnt += 1
                        old_state_dict[k] = new_state_dict[k]
                for k, v in self.model.named_parameters():
                    if 'visual_front_encoder' in k:
                        # print(k)
                        v.requires_grad = False
                print('\nload', cnt, 'keys from:%s\n' % self.args.model_state_dict_path)
                self.model.load_state_dict(old_state_dict)
            else:
                if self.args.device == 'gpu':
                    self.model.load_state_dict(torch.load(self.args.model_state_dict_path), strict=True)
                elif self.args.device == 'cpu':
                    m = torch.load(self.args.model_state_dict_path)
                    m = {k.replace('module.', ''): v for k, v in m.items()}
                    self.model.load_state_dict(m, strict=False)

                print('\nload from:%s\n' % self.args.model_state_dict_path)

    def _build_optimizer(self):
        from optimizer.adam_optimizer import AdamOptimizer
        from optimizer.lr_scheduler.inverse_square_root_schedule import InverseSquareRootSchedule
        from optimizer.lr_scheduler.cosine_lr_scheduler import CosineSchedule

        l_p = list(self.model.parameters())
        self.optimizer = AdamOptimizer(self.args, l_p)
        if self.args.lr_scheduler == 'cosine':
            self.lr_scheduler = CosineSchedule(self.args, self.optimizer)
        elif self.args.lr_scheduler == 'inverse_sqrt':
            self.lr_scheduler = InverseSquareRootSchedule(self.args, self.optimizer)
