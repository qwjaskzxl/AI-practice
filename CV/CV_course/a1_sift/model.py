import cv2
import numpy as np
from a1_sift.SIFT import SIFT_
from scipy.cluster.vq import kmeans, vq
from sklearn.metrics.pairwise import cosine_similarity


# import pylab

class RecogModel:
    def __init__(self, data):
        self.data = data
        # print('image list:', data.img)

    def get_sift(self, img):
        sift_det = cv2.xfeatures2d.SIFT_create()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, img_des = sift_det.detectAndCompute(gray, None)  # kp是这幅图像的所有keypoints，是内存地址一样的东西
        # img_des是这幅图像的keypoint descriptor的列表，shape为(num of sift, 128)。即第一幅图像中提取到了3765个关键点，每个关键点对应一个描述符，而一个描述符是一个128维的向量
        return img_des

    def k_means(self, feat, k):
        vocab, variance = kmeans(feat, k, 1)
        # print(vocab.shape, variance)
        # vocabulary是聚类中心列表，shape为(1000, 128)。
        return vocab

    def calculate_histogram(self, feat):  # 计算特征直方图向量
        img_words_labels, distance = vq(feat, self.vocab)  # 使用vq函数根据聚类中心对图像的所有sift descriptors进行分类，就像贴标签
        return img_words_labels

    def test(self):
        precision, recall = 0, 0
        precision_topk, recall_topk = 0, 0
        k = 5
        for i, file in enumerate(self.data.img):
            img = cv2.imread(file)
            feat = self.get_sift(img)
            img_hist = np.zeros((1, len(self.vocab)), "float32")
            img_words_labels = self.calculate_histogram(feat)
            for label in img_words_labels:  # 图像1有num of sift个label。 label∈[0,1000)
                img_hist[0][label] += 1

            similarity = cosine_similarity(img_hist, self.img_hist)
            most_similar_idx = np.argsort(similarity).reshape(-1)[::-1]
            for j in range(1, 1 + k):
                if self.data.img2label[self.data.img[most_similar_idx[j]]] == self.data.label[i]:
                    precision_topk += 1
                    if j == 1 :
                        precision += 1
                    break
        precision /= len(self.data.img)
        precision_topk /= len(self.data.img)
        print('top1 precision:%.3f'%precision,'top%d precision:%.3f' %(k,precision_topk),sep='\n')

    def build_model(self):
        feat_list = []
        for file in self.data.img:
            # print(file)
            img = cv2.imread(file)  # darray
            # self.sift = SIFT(img)
            # self.sift.get_DoG()
            feat = self.get_sift(img)
            feat_list.append(feat)
        feat = feat_list[0]
        for f in feat_list[1:]:
            feat = np.concatenate((feat, f), axis=0)
        print('whole features\' shape:', feat.shape)

        self.vocab = self.k_means(feat, k=1024)

        self.img_hist = np.zeros((len(feat_list), len(self.vocab)), "float32")
        for i in range(len(feat_list)):
            img_words_labels = self.calculate_histogram(feat_list[i])
            for label in img_words_labels:  # 图像1有num of sift个label。 label∈[0,1000)
                self.img_hist[i][label] += 1
        print('hist\'s shape:',self.img_hist.shape)


if __name__ == '__main__':
    pass
