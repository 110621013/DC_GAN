import os
import numpy as np
from PIL import Image

length, width = 538, 1000

def originImg2npy():
    from_path = 'C:/Users/user/Desktop/myself/original_dataset/'
    to_path = 'C:/Users/user/Desktop/GAN/data/npydata/'
    test_or_train = ['test-images', 'training-images']
    one_or_zero = ['0', '1']

    for tot in test_or_train:
        #為了取亂數0/1而用dict搗亂資料固定的0/1格式(0/1資料亂數排序)
        s = {}
        for ooz in one_or_zero:
            print(tot, ooz)
            file_source_path = from_path+tot+'/'+ooz+'/'
            img_list = os.listdir(file_source_path)

            for i in range(len(img_list)):
                img_name = img_list[i]
                s[img_name] = ooz

        i = 0
        for img_name, ooz in s.items():
            all_img_array = np.empty((len(s), length, width), dtype=int)
            all_label_array = np.empty(len(s), dtype=int)

            img = Image.open(from_path+tot+'/'+ooz+'/'+img_name).convert('L')
            img_array = np.array(img)
            all_img_array[i] = img_array

            all_label_array[i] = int(ooz)

            i+=1

        np.save(to_path+tot.split('-')[0]+'data', all_img_array)
        np.save(to_path+tot.split('-')[0]+'label', all_label_array)



def check():
    to_path = 'C:/Users/user/Desktop/GAN/data/npydata/'
    t = np.load(to_path+'trainingdata.npy')
    print(t.shape)
    t = np.load(to_path+'traininglabel.npy')
    print(t.shape)
    t = np.load(to_path+'testdata.npy')
    print(t.shape)
    t = np.load(to_path+'testlabel.npy')
    print(t.shape)

if __name__ == '__main__':
    #originImg2npy()
    check()