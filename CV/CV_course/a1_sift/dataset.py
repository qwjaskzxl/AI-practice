import os, json


class VGGDataset:
    def __init__(self, root, **kwargs):
        self.root = root
        self.scene = os.listdir(root)
        # self._EDA()
        self.get_img(**kwargs)

    def get_img(self, **kwargs) -> dict:
        img2label = {}
        category = set()  # name
        if 'scene_choice' in kwargs:
            print('scene_choice:', kwargs['scene_choice'])
            s = kwargs['scene_choice']
            path_s = os.path.join(self.root, s)
            for name in os.listdir(path_s):
                path_img = path_s + '/' + name
                if len(os.listdir(path_img)) > 5:
                    category.add(name)
                    for img in os.listdir(path_img):
                        img2label[path_img + '/' + img] = name
        self.img2label = img2label
        self.category = category
        self.label = list(img2label.values())
        print('num of image:', len(img2label))
        print('num of category:', len(category))
        return img2label

    def _EDA(self):
        for s in self.scene:
            names = set()
            path = os.path.join(self.root, s)
            for n in os.listdir(path):
                if len(os.listdir(path + '/' + n)) > 3:
                    names.add(n)
            print(s, len(names))
        '''
        >2:
        football 288
        stage 1142
        conference_room 10
        boat 231
        ice_skating 76
        coffee_shop 4
        office 84
        hospital 73
        kitchen 75
        staircase 28
        supermarket 49
        golf 147
        banquet 0
        beach 760
        desert 3
        airport_terminal 28
        '''

    def _bulid_label2id(self):
        d = {l: i for i, l in enumerate(self.label)}
        with open('data/label2id.json', 'w') as f:
            json.dump(d, f)
        print(d)

    @property
    def img(self):
        return list(self.img2label)


if __name__ == '__main__':
    data = VGGDataset('/home1/lihaoyuan/data/CV/vgg_celebrity_dataset/Image/Images', scene_choice='stage')
