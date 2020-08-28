import os
import numpy as np
from PIL import Image

length, width = 28, 28
train_ratio = 0.7

def originImg2npy():
    from_path = 'C:\\Users\\user\\Desktop\\radar\\'
    to_path = 'C:\\Users\\user\\Desktop\\GAN\\data\\radar\\'

    img_path_list = []
    for root, subdirs, files in os.walk(from_path):
        for f in files:
            if f.endswith('.png'):img_path_list.append(os.path.join(root, f))
    print(len(img_path_list), img_path_list[0], img_path_list[-1])

    train_num = int(len(img_path_list)*train_ratio)
    test_num = len(img_path_list)-train_num

    train_img_array = np.empty((train_num, length, width), dtype=int)
    test_img_array = np.empty((test_num, length, width), dtype=int)
    for i in range(len(img_path_list)):
        try:
            img = Image.open(img_path_list[i]).convert('L')
        except:
            print('oof', img_path_list[i])
            img = Image.open(img_path_list[i-1]).convert('L')
        img_array = np.array(img)
        if i < train_num:train_img_array[i] = img_array
        else:test_img_array[i-train_num] = img_array

    np.save(to_path + 'train.npy', train_img_array)
    np.save(to_path + 'test.npy', test_img_array)


def check():
    to_path = 'C:\\Users\\user\\Desktop\\GAN\\data\\radar\\'
    t = np.load(to_path+'train.npy')
    print(t.shape)
    t = np.load(to_path+'test.npy')
    print(t.shape)


if __name__ == '__main__':
    originImg2npy()
    check()