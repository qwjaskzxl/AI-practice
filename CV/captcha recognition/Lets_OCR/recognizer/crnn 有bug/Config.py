import alphabets
import os

# print(os.getcwd() )
raw_folder = ''
train_data = r'data/data/crnn_train_lmdb'
test_data = r'data/data/crnn_train2_lmdb'
random_sample = True
random_seed = 1111
using_cuda = True
keep_ratio = False
gpu_id = '2'
model_dir = 'w160_bs64_model_ywj_v1'
data_worker = 8
batch_size = 64
img_height = 100#32
img_width = 200#100
alphabet = alphabets.alphabet
epoch = 5
display_interval = 20
save_interval = 30
test_interval = 20
test_disp = 20
test_batch_num = 32
lr = 0.000001
beta1 = 0.5
infer_img_w = 160
