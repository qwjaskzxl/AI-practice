# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta

MAX_PREQ_N = 10
MIN_FREQ = 100  # 总共6979，1:6741, 5：5928/all:6038
reset_vocab = 0
MAX_VOCAB_SIZE = 200000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            content = tokenizer(content)

            bigram = []
            trigram = []
            for i in range(len(content[:])):
                try:
                    bigram.append(str(content[i]) + '_' + str(content[i + 1]))
                    trigram.append(str(content[i]) + '_' + str(content[i + 1]) + '_' + str(content[i + 2]))
                except:
                    break
            content += bigram + trigram

            for word in content:
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq],
                            key=lambda x: x[1], reverse=True)[MAX_PREQ_N:max_size + MAX_PREQ_N]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path) and not reset_vocab:
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=MIN_FREQ)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def _build_ngram_dataset():
        for path in [config.train_path, config.dev_path, config.test_path]:
            ngram_corpus = []
            with open(path, 'r', encoding='UTF-8') as f:
                for line in tqdm(f):
                    lin = line.strip()
                    if not lin:
                        continue
                    if 'test' in path:
                        content, label = lin, -1
                    else:
                        content, label = lin.split('\t')

                    content = tokenizer(content)
                    bigram = []
                    trigram = []
                    for i in range(len(content[:])):
                        try:
                            bigram.append(str(content[i]) + '_' + str(content[i + 1]))
                            trigram.append(str(content[i]) + '_' + str(content[i + 1]) + '_' + str(content[i + 2]))
                        except:
                            break
                    content += bigram + trigram
                    if 'test' in path:
                        ngram_corpus.append(' '.join(content) + '\n')
                    else:
                        ngram_corpus.append(' '.join(content) + '\t' + label + '\n')

                with open(path.split('.')[0] + '_ngram.txt', 'w') as f:
                    for c in ngram_corpus:
                        f.write(c)

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path.split('.')[0] + '_ngram.txt', 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                # for i,w in enumerate(lin[::-1]):
                #     if w >= '0' and w<='9':
                #         label = int(w)
                #         break
                # content = lin[:-2]
                if 'test' in path:
                    content, label = lin, -1
                else:
                    content, label = lin.split('\t')

                content = tokenizer(content)
                # bigram = []
                # trigram = []
                # for i in range(len(content[:])):
                #     try:
                #         bigram.append(str(content[i]) + '_' + str(content[i + 1]))
                #         trigram.append(str(content[i]) + '_' + str(content[i + 1]) + '_' + str(content[i + 2]))
                #     except:
                #         break
                # content += bigram + trigram
                # if 'test' in path:
                #     ngram_corpus.append(' '.join(content) + '\n')
                # else:
                #     ngram_corpus.append(' '.join(content) + '\t' + label + '\n')

                # if pad_size:
                #     if len(token) < pad_size:
                #         token.extend([PAD] * (pad_size - len(token)))
                #     else:
                #         token = token[:pad_size]
                #         seq_len = pad_size
                # words_line = []
                # for word in token:
                #     words_line.append(vocab.get(word, vocab.get(UNK)))

                words_line = [vocab[w] for w in content if w in vocab]
                seq_len = len(words_line)
                if pad_size:
                    if len(words_line) < pad_size:
                        words_line.extend([vocab[PAD]] * (pad_size - len(words_line)))
                    else:
                        words_line = words_line[:pad_size]
                        seq_len = pad_size
                contents.append((words_line, int(label), seq_len))

            # with open(path.split('.')[0] + '_ngram.txt', 'w') as f:
            #     for c in ngram_corpus:
            #         f.write(c)
        return contents  # [([...], 0), ([...], 1), ...]

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
