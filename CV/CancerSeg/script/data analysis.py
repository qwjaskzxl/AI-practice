import os, json, random
import pandas as pd

root = '/home1/lihaoyuan/data/CV/CancerSeg/data/'
with open(root + '/train/train_label.csv') as f:
    train_label = pd.read_csv(f)
# print(os.listdir(root))
'''
train_org_image
train_mask
train_label.csv
'''

print(train_label.info())
print(train_label['label'].value_counts())

cate_freq = {}
for name, count in train_label['label'].value_counts().items():
    cate_freq[name] = count
id2cate = {}


def build_id2cate():
    for i in range(len(train_label)):
        row = train_label.iloc[i]
        id2cate[row['image_name']] = int(row['label'])
    print(id2cate)
    with open(root + 'id2cate.json', 'w') as f:
        json.dump(id2cate, f)


def split_train_val():
    train_lists = os.listdir(root + '/train/train_org_image/')
    train_list, val_list = [], []
    cate_freq_now = {n: 0 for n in range(4)}
    random.shuffle(train_lists)
    for x in train_lists:
        if 'zYogd' in x:
            continue

        C = id2cate[x]
        path = root + 'train/train_org_image/' + x
        if cate_freq_now[C] >= int(0.8 * cate_freq[C]):
            val_list.append(path)
        else:
            train_list.append(path)
            cate_freq_now[C] += 1
    D = {}
    for x in val_list:
        x = x.split('/')[-1]
        if id2cate[x] in D:
            D[id2cate[x]] += 1
        else:
            D[id2cate[x]] = 1
    print(D)
    with open(root + 'train_list.json', 'w') as f1, open(root + 'val_list.json', 'w') as f2:
        json.dump(train_list, f1)
        json.dump(val_list, f2)


build_id2cate()
split_train_val()
