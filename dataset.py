from utils import DataProcess
import numpy as np
import random


class Generator:
    def __init__(self, label_file, img_w, img_h,
                 batch_size, downsample_factor):
        self.process = DataProcess(label_file,img_w, img_h)
        self.process.get_data()
        self.process.pad()
        self.process.preprocess_img()
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.class_num = self.process.class_num
        self.max_len = self.process.max_len
        self.downsample_factor = downsample_factor
        self.n = len(self.process.img_paths)                      # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.texts = self.process.padded_txt
        self.imgs = self.process.imgs

    def next_sample(self):     
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]


    def next_batch(self):   
        while True:
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])     # (bs, 128, 64, 1)
            Y_data = np.ones([self.batch_size, self.max_len])             # (bs, 9)
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1))           # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = text
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,  # (bs, 128, 32, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  
                'label_length': label_length  
            }
            outputs = {'ctc': np.zeros([self.batch_size])}   
            yield (inputs, outputs)  



if __name__ == '__main__':
    gen = Generator('data/recognition/train_label.txt',32,128,4,4)
    inputs,o = gen.next_batch()
    x,y,in_len,out_len = list(inputs.values())
    print(x.shape)
    print(y.shape)
    print(in_len)
    print(out_len)

