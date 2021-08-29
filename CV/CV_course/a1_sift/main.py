# import warnings
# warnings.filterwarnings("ignore")
from a1_sift.dataset import VGGDataset
from a1_sift.model import RecogModel

if __name__ == '__main__':
    data = VGGDataset('/home1/lihaoyuan/data/CV/vgg_celebrity_dataset/Image/Images',
                      scene_choice='supermarket')
    model = RecogModel(data)
    model.build_model()
    model.test()

'''
优化：
1. 把相同尺寸的图片放一起/resize到相同尺寸，batch处理
2. 多线程
'''