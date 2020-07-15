


import lmdb
import base64
import six
from PIL import Image
import numpy as np
import chardet

import cv2

train_data = r'D:\graduate_design\data\data\crnn_train2_lmdb'
env = lmdb.open(train_data)

txn = env.begin(write=False)

index = 10
img_key = 'image-%09d' % index
imgbuf = txn.get(img_key.encode())
print(imgbuf)
print(chardet.detect(imgbuf))
imgbuf = base64.b64decode(imgbuf)  # add 20200626
print(imgbuf)
buf = six.BytesIO()
buf.write(imgbuf)
buf.seek(0)



# img = Image.open(r'D:\比赛\验证码识别\test\0.jpg',)#.convert('L')
# img = img.convert("YCbCr")
img = Image.open(buf).convert('L')
img = img.resize((100, 32), Image.ANTIALIAS)
#img.show()
width = img.size[0]
height = img.size[1]
print(width,height)
# img_data = np.array(img)
array = []
for x in range(width):
    for y in range(height):
        r = img.getpixel((x,y))
        # gb = (r,g,b)
        array.append(r)
arr1 = np.array(array).reshape((100,32,1))
#arr1 = arr1.T
arr1=np.fliplr(arr1)
arr1 = np.rot90(arr1)
img = Image.fromarray(arr1.astype( np.uint8 ))
img.show()
print(arr1[:,:,0][0])



# mode_list = ['1', 'L', 'I', 'F', 'P', 'RGB', 'RGBA', 'CMYK', 'YCbCr' ]
# for mode in mode_list:
#     try:
#         img = img.convert(mode)
#         img_data = np.array(img)
#         print(mode,img_data.shape,img_data)
#         print('---')
#     except:
#         pass