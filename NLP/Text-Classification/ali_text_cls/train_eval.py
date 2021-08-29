# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    dev_best_f1 = 0.8
    last_improve = 0  # 记录上次验证集f1提升的batch数
    flag = False  # 记录是否很久没有效果提升

    # 正向RNN TextRNN_0.996893_59127.ckpt
    # 反向RNN TextRNN_0.995728_83073.ckpt
    # 正向CNN TextCNN.ckpt_0.995804_01942.ckpt
    # 反向CNN TextCNN.ckpt_0.990660_91918.ckpt
    # generate_any_result(config, model, test_iter, 'dataset/saved_dict/TextCNN.ckpt_0.995804_01942.ckpt')
    # exit()

    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            model.train()
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                train_f1 = f1_score(true, predic, average='macro')

                dev_loss, dev_acc, dev_f1 = evaluate(config, model, dev_iter)

                if dev_f1 > dev_best_f1:
                    dev_best_f1 = dev_f1
                    best_model_path = config.save_path + '_%4f_%s.ckpt' % (dev_f1, str(time.time())[-5:])
                    improve = '*'
                    last_improve = total_batch
                    if 'train' in config.train_path:
                        torch.save(model.state_dict(), best_model_path)
                        generate_result(config, model, test_iter, best_model_path, dev_best_f1)

                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter:{0}  Train Loss:{1:.2}  Train F1:{2:.4}   Val Loss:{3:.2}  Val F1:{4:.4}  Time:{5} {6}'
                print(msg.format(total_batch, loss.item(), train_f1, dev_loss, dev_f1, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                writer.add_scalar("f1/train", dev_loss, total_batch)
                writer.add_scalar("f1/dev", dev_loss, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

        dev_loss, _, dev_f1 = evaluate(config, model, dev_iter, an_epoch=True)
        if dev_f1 > dev_best_f1:
            dev_best_f1 = dev_f1
            best_model_path = config.save_path + '_%4f_%s.ckpt' % (dev_f1, str(time.time())[-5:])
            improve = '*'
            last_improve = total_batch
            if 'train' in config.train_path:
                torch.save(model.state_dict(), best_model_path)
                generate_result(config, model, test_iter, best_model_path, dev_best_f1)
        else:
            improve = ''

        msg = 'Epoch:{0}   Val Loss:{1:.2}  Val F1:{2:.4}  {3}'
        print(msg.format(epoch, dev_loss, dev_f1, improve))

    writer.close()


def generate_any_result(config, model, test_iter, best_model_path):
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    output_all = np.zeros((0, 14), dtype=float)
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in test_iter:
            output = model(texts)
            output_all = np.append(output_all, output.data.cpu().numpy(), axis=0)
            predict = torch.max(output.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predict)
            labels = labels.data.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            print(f1_score(labels, predict, average='macro'))

    f1 = f1_score(labels_all, predict_all, average='macro')

    with open('results/probs/%s_result_%.4f.npy' % (config.model_name, f1), 'wb') as f:
        np.save(f, output_all)


def generate_result(config, model, test_iter, best_model_path, dev_f1):
    # test
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    predict_all = np.array([], dtype=int)
    output_all = np.zeros((0, 14), dtype=float)
    with torch.no_grad():
        for texts, labels in test_iter:
            output = model(texts)
            output_all = np.append(output_all, output.data.cpu().numpy(), axis=0)
            predic = torch.max(output.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)

    with open('results/probs/%s_result_%.4f.npy' % (config.model_name, dev_f1), 'wb') as f:
        np.save(f, output_all)

    with open('results/%s_result_%.4f.csv' % (config.model_name, dev_f1), 'w') as f:
        f.write('label\n')
        for p in predict_all:
            f.write(str(p) + '\n')


def evaluate(config, model, data_iter, an_epoch=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    f1 = f1_score(labels_all, predict_all, average='macro')
    if an_epoch:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        # report默认的average = 'weighted'
        print("Precision, Recall and F1-Score...")
        print(report)
        # confusion = metrics.confusion_matrix(labels_all, predict_all)
        # print("Confusion Matrix...")
        # print(confusion)

    return loss_total / len(data_iter), acc, f1


if __name__ == '__main__':
    with open('results/probs/Transformer_result_0.0267', 'rb') as f:
        a = np.load(f)
        print(a)
