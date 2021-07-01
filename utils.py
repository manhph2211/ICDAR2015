import string
import numpy as np 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2
import json


class DataProcess:

	def __init__(self,label_file,img_w, img_h):
		self.label_file = label_file
		self.char_list = ''
		self.img_w = img_w
		self.img_h = img_h


	def pad(self):
		for txt in self.txts:
			for char in txt:
				if char not in self.char_list:
					self.char_list += char
		self.char_list = ''.join(sorted(self.char_list))
		self.c2i = {c:i for i,c in enumerate(self.char_list)}
		self.i2c = {i:c for i,c in enumerate(self.char_list)}
		self.class_num = len(self.c2i) + 1
		encoded_label = [self.encode(text) for text in self.txts]
		self.padded_txt = pad_sequences(encoded_label, maxlen=self.max_len, padding='post', value = len(self.char_list))


	def encode(self,txt):
		return [self.c2i[char] for char in txt]


	def get_data(self):
		self.max_len = 0
		self.samples = {}
		with open(self.label_file,'r') as f:
			data = f.readlines()
			for line in data:
				line = line.split()
				line[1] = line[1].replace('\n','')
				self.samples['data/recognition/'+line[0]] = line[1]
				if len(line[1]) > self.max_len:
					self.max_len = len(line[1])
		# write_json(self.save_file,samples)
		self.txts = list(self.samples.values())
		self.img_paths = list(self.samples.keys())
		self.imgs = np.zeros((len(self.img_paths), self.img_h, self.img_w))


	def preprocess_img(self):
		for i,img_file in enumerate(self.img_paths):
			img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(img, (self.img_w, self.img_h))
			img = img.astype(np.float32)
			img = (img / 255.0) * 2.0 - 1.0
			self.imgs[i, :, :] = img


def read_json(file):
	with open(file,'r') as f:
		data = json.load(f)
	return data


def write_json(file,data):
	with open(file,'w') as f:
		json.dump(data,f,indent = 4)


def remove_dup(list_idx):
  text=[0]
  list_idx=list_idx[0]
  for i in range(len(list_idx)-1):
    if list_idx[i]== text[-1] :
        continue
    text.append(list_idx[i])
  return text


def decode(list_idx,vocab):
  text=remove_dup(list_idx)
  text=[x for x in text if x !=0]
  text=[vocab[idx-1] for idx in text]
  return ''.join(text)


if __name__ == '__main__':
	test_label_file = 'data/recognition/test_label.txt'
	train_label_file = 'data/recognition/train_label.txt'
	adu =DataProcess(train_label_file,28,120)
	adu.get_data()
	adu.pad()
	adu.preprocess_img()
	print(adu.char_list)
	print(adu.class_num)
	print(adu.max_len)
