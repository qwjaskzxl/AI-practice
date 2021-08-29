# 导入相关包
import copy
import os
import random
import numpy as np
import jieba as jb
import jieba.analyse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
from torchtext.data import Field
from torchtext.data import Dataset
from torchtext.data import Iterator
from torchtext.data import Example
from torchtext.data import BucketIterator

# from tensorboardX import SummaryWriter
# writer = SummaryWriter('./tensorboard')

dataset = {}
path = "dataset/"
files = os.listdir(path)
for file in files:
    if not os.path.isdir(file) and not file[0] == '.':  # 跳过隐藏文件和文件夹
        f = open(path + "/" + file, 'r', encoding='UTF-8');  # 打开文件
        for line in f.readlines():
            dataset[line] = file[:-4]

name_zh = {'LX': '鲁迅', 'MY': '莫言', 'QZS': '钱钟书', 'WXB': '王小波', 'ZAL': '张爱玲'}
for (k, v) in list(dataset.items())[:6]:
    print(k, '---', name_zh[v])

# 精确模式分词
titles = [".".join(jb.cut(t, cut_all=False)) for t, _ in dataset.items()]
print("精确模式分词结果:\n", titles[0])

# 全模式分词
titles = [".".join(jb.cut(t, cut_all=True)) for t, _ in dataset.items()]
print("全模式分词结果:\n", titles[0])

# 搜索引擎模式分词
titles = [".".join(jb.cut_for_search(t)) for t, _ in dataset.items()]
print("搜索引擎模式分词结果:\n", titles[0])

# 将片段进行词频统计
str_full = {}
str_full['LX'] = ""
str_full['MY'] = ""
str_full['QZS'] = ""
str_full['WXB'] = ""
str_full['ZAL'] = ""

for (k, v) in dataset.items():
    str_full[v] += k

for (k, v) in str_full.items():
    print(k, ":")
    for x, w in jb.analyse.extract_tags(v, topK=5, withWeight=True):
        print('%s %s' % (x, w))


def load_data(path):
    """
    读取数据和标签
    :param path:数据集文件夹路径
    :return:返回读取的片段和对应的标签
    """
    sentences = []  # 片段
    target = []  # 作者

    # 定义lebel到数字的映射关系
    labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

    files = os.listdir(path)
    for file in files:
        if not os.path.isdir(file):
            f = open(path + "/" + file, 'r', encoding='UTF-8');  # 打开文件
            for index, line in enumerate(f.readlines()):
                sentences.append(line)
                target.append(labels[file[:-4]])

    return list(zip(sentences, target))


# 定义Field
TEXT = Field(sequential=True, tokenize=lambda x: jb.lcut(x), lower=True, use_vocab=True)
LABEL = Field(sequential=False, use_vocab=False)
FIELDS = [('text', TEXT), ('category', LABEL)]

# 读取数据，是由tuple组成的列表形式
mydata = load_data(path)

# 使用Example构建Dataset
examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), mydata))
dataset = Dataset(examples, fields=FIELDS)
# 构建中文词汇表
TEXT.build_vocab(dataset)
print('词数:', len(TEXT.vocab))
# 切分数据集
train, val = dataset.split(split_ratio=0.9)

# 生成可迭代的mini-batch
train_iter, val_iter = BucketIterator.splits(
    (train, val),  # 数据集
    batch_sizes=(8, 256),
    device=0,  # 如果使用gpu，此处将-1更换为GPU的编号
    sort_key=lambda x: len(x.text),
    sort_within_batch=False,
    repeat=False
)


# Pytorch定义模型的方式之一：
# 继承 Module 类并实现其中的forward方法
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.emb = nn.Embedding(num_embeddings=len(TEXT.vocab), embedding_dim=300)
#         self.lstm = torch.nn.LSTM(300, 128, bidirectional=True, dropout=0.2, batch_first=True)
#         filter_sizes = (1, 2, 3, 4, 5)
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(1, 128, (k, 300)) for k in filter_sizes])
#
#         self.fc_1 = nn.Linear(128 * len(filter_sizes), 5)
#         self.fc_2 = nn.Linear(128, 5)
#
#         self.fc1 = nn.Linear(256, 128)
#         self.fc2 = nn.Linear(128, 5)
#         self.drop = nn.Dropout(0.1)
#
#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)).squeeze(3)
#         x = F.max_pool1d(x, int(x.size(2))).squeeze(2)
#         return x
#
#     def forward(self, x):
#         """
#         前向传播
#         :param x: 模型输入
#         :return: 模型输出
#         """
#         x = self.emb(x).permute(1, 0, 2)
#         x_ = torch.cat([self.conv_and_pool(x.unsqueeze(1), conv) for conv in self.convs], 1)
#         x_ = self.fc_1(x_)
#         # x_ = self.drop(x_)
#         # x_ = self.fc_2(x_)
#
#         output, hidden = self.lstm(x)
#         # print(hidden[0].shape)
#         # x = output[:, 0, :]
#         x = output.mean(1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         out = x + x_
#         return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.emb = nn.Embedding(num_embeddings=len(TEXT.vocab), embedding_dim=300)
        self.lstm = torch.nn.LSTM(300, 128, bidirectional=True, dropout=0.2, batch_first=True)
        self.fc1 = nn.Linear(256, 5)
        self.fc2 = nn.Linear(128, 5)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.emb(x).permute(1, 0, 2)
        output, hidden = self.lstm(x)
        # print(hidden[0].shape)
        # x = output[:, 0, :]
        x = output.mean(1)
        x = self.fc1(x)
        # x = self.fc2(x)
        out = x
        return out


# 创建模型实例
device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")

model = Net().to(device)

for name, parameters in model.named_parameters():
    print(name, ':', parameters.size())

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

train_acc_list, train_loss_list = [], []
val_acc_list, val_loss_list = [], []

i = 0
for epoch in range(5):
    train_acc, train_loss = 0, 0
    for idx, batch in enumerate(train_iter):
        text, label = batch.text, batch.category
        # print(text, label)
        # exit()
        text = text.to(device)

        # writer.add_graph(model, input_to_model=text)
        # exit()
        label = label.to(device)
        optimizer.zero_grad()
        out = model(text)
        loss = loss_fn(out, label.long())
        loss.backward(retain_graph=True)
        optimizer.step()
        accracy = np.mean((torch.argmax(out, 1) == label).cpu().numpy())
        # 计算每个样本的acc和loss之和
        train_acc += accracy * len(batch)
        train_loss += loss.item() * len(batch)

        print("\r opech:{} loss:{}, train_acc:{}".format(epoch, loss.item(), accracy), end=' ')

        # 在验证集上预测
    #     val_acc, val_loss = 0, 0
    #     with torch.no_grad():
    #         for idx, batch in enumerate(val_iter):
    #             text, label = batch.text, batch.category
    #             text = text.to(device)
    #             label = label.to(device)
    #             out = model(text)
    #             loss = loss_fn(out, label.long())
    #             accracy = np.mean((torch.argmax(out, 1) == label).cpu().numpy())
    #             # 计算一个batch内每个样本的acc和loss之和
    #             val_acc += accracy * len(batch)
    #             val_loss += loss.item() * len(batch)
    #         val_acc /= len(val_iter.dataset)
    #         writer.add_histogram('val_acc', val_acc, i)
    #         i += 1
    #
    #
    # train_acc /= len(train_iter.dataset)
    # train_loss /= len(train_iter.dataset)
    # val_acc /= len(val_iter.dataset)
    # val_loss /= len(val_iter.dataset)
    # train_acc_list.append(train_acc)
    # train_loss_list.append(train_loss)
    # val_acc_list.append(val_acc)
    # val_loss_list.append(val_loss)
    # print('final val_acc:', np.array(val_acc_list).mean())
# 保存模型
torch.save(model.state_dict(), 'results/temp.pth', _use_new_zipfile_serialization=False)

# # 绘制曲线
# plt.figure(figsize=(15, 5.5))
# plt.subplot(121)
# plt.plot(train_acc_list)
# plt.plot(val_acc_list)
# plt.title("acc")
# plt.subplot(122)
# plt.plot(train_loss_list)
# plt.plot(val_loss_list)
# plt.title("loss")


model_path = "results/temp0.7.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


# val_acc, val_loss = 0, 0

# with torch.no_grad():
#     for idx, batch in enumerate(val_iter):
#         text, label = batch.text, batch.category
#         text = text.to(device)
#         label = label.to(device)
#         out = model(text)
#         loss = loss_fn(out, label.long())
#         accracy = np.mean((torch.argmax(out, 1) == label).cpu().numpy())
#         # 计算一个batch内每个样本的acc和loss之和
#         val_acc += accracy * len(batch)
#     val_acc /= len(val_iter.dataset)
#     print(val_acc)


# print('模型加载完成...')
#
# # 这是一个片段
# text = "中国中流的家庭，教孩子大抵只有两种法。其一是任其跋扈，一点也不管，\
#     骂人固可，打人亦无不可，在门内或门前是暴主，是霸王，但到外面便如失了网的蜘蛛一般，\
#     立刻毫无能力。其二，是终日给以冷遇或呵斥，甚于打扑，使他畏葸退缩，彷佛一个奴才，\
#     一个傀儡，然而父母却美其名曰“听话”，自以为是教育的成功，待到他们外面来，则如暂出樊笼的\
#     小禽，他决不会飞鸣，也不会跳跃。"
#
# labels = {0: '鲁迅', 1: '莫言', 2: '钱钟书', 3: '王小波', 4: '张爱玲'}
#
# # 将句子做分词，然后使用词典将词语映射到他的编号
# text2idx = [TEXT.vocab.stoi[i] for i in jb.lcut(text)]
#
# # 转化为Torch接收的Tensor类型
# text2idx = torch.Tensor(text2idx).long()
#
# # 预测
# print(labels[torch.argmax(model(text2idx.view(-1, 1)), 1).numpy()[0]])


# 导入相关包
# import copy
# import os
# import numpy as np
# import jieba as jb
# import torch
# import torch.nn as nn
# import torch.nn.functional as f
#
# from torchtext import data, datasets
# from torchtext.data import Field,Dataset,Iterator,Example,BucketIterator

# class Net(nn.Module):
#     def __init__(self,vocab_size):
#         super(Net,self).__init__()
#         pass
#
#     def forward(self,x):
#         """
#         前向传播
#         :param x: 模型输入
#         :return: 模型输出
#         """
#         pass

def processing_data(data_path, split_ratio=0.7):
    """
    数据处理
    :data_path：数据集路径
    :validation_split：划分为验证集的比重
    :return：train_iter,val_iter,TEXT.vocab 训练集、验证集和词典
    """
    # --------------- 已经实现好数据的读取，返回和训练集、验证集，可以根据需要自行修改函数 ------------------
    sentences = []  # 片段
    target = []  # 作者

    # 定义lebel到数字的映射关系
    labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

    files = os.listdir(data_path)
    for file in files:
        if not os.path.isdir(file):
            f = open(data_path + "/" + file, 'r', encoding='UTF-8');  # 打开文件
            for index, line in enumerate(f.readlines()):
                sentences.append(line)
                target.append(labels[file[:-4]])

    mydata = list(zip(sentences, target))

    TEXT = Field(sequential=True, tokenize=lambda x: jb.lcut(x),
                 lower=True, use_vocab=True)
    LABEL = Field(sequential=False, use_vocab=False)

    FIELDS = [('text', TEXT), ('category', LABEL)]

    examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS),
                        mydata))

    dataset = Dataset(examples, fields=FIELDS)

    TEXT.build_vocab(dataset)

    train, val = dataset.split(split_ratio=split_ratio)

    # BucketIterator可以针对文本长度产生batch，有利于训练
    train_iter, val_iter = BucketIterator.splits(
        (train, val),  # 数据集
        batch_sizes=(16, 16),
        device=1,  # 如果使用gpu，此处将-1更换为GPU的编号
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        repeat=False  #
    )
    # --------------------------------------------------------------------------------------------
    return train_iter, val_iter, TEXT.vocab


# def model(train_iter, val_iter, Text_vocab,save_model_path):
#     """
#     创建、训练和保存深度学习模型
#
#     """
#     # --------------------- 实现模型创建、训练和保存等部分的代码 ---------------------
#     pass
#     # 保存模型（请写好保存模型的路径及名称）
#
#     # --------------------------------------------------------------------------------------------
#
# def evaluate_mode(val_iter, save_model_path):
#     """
#     加载模型和评估模型
#     可以实现，比如: 模型训练过程中的学习曲线，测试集数据的loss值、准确率及混淆矩阵等评价指标！
#     主要步骤:
#         1.加载模型(请填写你训练好的最佳模型),
#         2.对自己训练的模型进行评估
#
#     :param val_iter: 测试集
#     :param save_model_path: 加载模型的路径和名称,请填写你认为最好的模型
#     :return:
#     """
#     # ----------------------- 实现模型加载和评估等部分的代码 -----------------------
#     pass
#
#     # ---------------------------------------------------------------------------
#
# def main():
#     """
#     深度学习模型训练流程,包含数据处理、创建模型、训练模型、模型保存、评价模型等。
#     如果对训练出来的模型不满意,你可以通过调整模型的参数等方法重新训练模型,直至训练出你满意的模型。
#     如果你对自己训练出来的模型非常满意,则可以提交作业!
#     :return:
#     """
#     data_path = "./dataset"  # 数据集路径
#     save_model_path = "results/model.pth"  # 保存模型路径和名称
#     train_val_split = 0.7 #验证集比重
#
#     # 获取数据、并进行预处理
#     train_iter, val_iter,Text_vocab = processing_data(data_path, split_ratio = train_val_split)
#
#     # 创建、训练和保存模型
#     model(train_iter, val_iter, Text_vocab, save_model_path)
#
#     # 评估模型
#     evaluate_mode(val_iter, save_model_path)
#
#
# if __name__ == '__main__':
#     main()

# ==================  提交 Notebook 训练模型结果数据处理参考示范  ==================
# 导入相关包


# ------------------------------------------------------------------------------
# 本 cell 代码仅为 Notebook 训练模型结果进行平台测试代码示范
# 可以实现个人数据处理的方式，平台测试通过即可提交代码
#  -----------------------------------------------------------------------------

def load_data(path):
    """
    读取数据和标签
    :param path:数据集文件夹路径
    :return:返回读取的片段和对应的标签
    """
    sentences = []  # 片段
    target = []  # 作者

    # 定义lebel到数字的映射关系
    labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

    files = os.listdir(path)
    for file in files:
        if not os.path.isdir(file):
            f = open(path + "/" + file, 'r', encoding='UTF-8');  # 打开文件
            for index, line in enumerate(f.readlines()):
                sentences.append(line)
                target.append(labels[file[:-4]])

    return list(zip(sentences, target))


# 定义Field
# TEXT  = Field(sequential=True, tokenize=lambda x: jb.lcut(x), lower=True, use_vocab=True)
# LABEL = Field(sequential=False, use_vocab=False)
# FIELDS = [('text', TEXT), ('category', LABEL)]
#
# # 读取数据，是由tuple组成的列表形式
# mydata = load_data(path='dataset/')
#
# # 使用Example构建Dataset
# examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), mydata))
# dataset = Dataset(examples, fields=FIELDS)
# # 构建中文词汇表
# TEXT.build_vocab(dataset)
#
# # =========================  Notebook 训练模型网络结构参考示范  =========================
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.lstm = torch.nn.LSTM(1,64)
#         self.fc1 = nn.Linear(64,128)
#         self.fc2 = nn.Linear(128,5)
#
#     def forward(self,x):
#         """
#         前向传播
#         :param x: 模型输入
#         :return: 模型输出
#         """
#         output,hidden = self.lstm(x.unsqueeze(2).float())
#         h_n = hidden[1]
#         out = self.fc2(self.fc1(h_n.view(h_n.shape[1],-1)))
#         return out
# #
# # ----------------------------- 请加载您最满意的模型 -------------------------------
# # 加载模型(请加载你认为的最佳模型)
# # 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
# # 如果你的模型是在 results 文件夹下的 temp.pth 模型，则 model_path = 'results/temp.pth'
#
# # 创建模型实例
# model = None
# model_path = None
# model.load_state_dict(torch.load(model_path))
#
# # -------------------------请勿修改 predict 函数的输入和输出-------------------------
def predict(text):
    """
    :param text: 中文字符串
    :return: 字符串格式的作者名缩写
    """
    # ----------- 实现预测部分的代码，以下样例可代码自行删除，实现自己的处理方式 -----------
    labels = {0: 'LX', 1: 'MY', 2: 'QZS', 3: 'WXB', 4: 'ZAL'}
    # 自行实现构建词汇表、词向量等操作
    # 将句子做分词，然后使用词典将词语映射到他的编号
    text2idx = [TEXT.vocab.stoi[i] for i in jb.lcut(text)]
    # 转化为Torch接收的Tensor类型
    text2idx = torch.Tensor(text2idx).long()

    # 模型预测部分
    results = model(text2idx.view(-1, 1))
    prediction = labels[torch.argmax(results, 1).numpy()[0]]
    # --------------------------------------------------------------------------

    return prediction
#
# sen = "我听到一声尖叫，感觉到蹄爪戳在了一个富有弹性的东西上。定睛一看，不由怒火中烧。原来，趁着我不在，隔壁那个野杂种——沂蒙山猪刁小三，正舒坦地趴在我的绣榻上睡觉。我的身体顿时痒了起来，我的目光顿时凶了起来。"
# predict(sen)
