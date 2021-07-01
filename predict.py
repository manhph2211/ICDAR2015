from model import get_Model
from config import *
import glob
import cv2
import numpy as np


def remove_dup(list_idx):
  text=[0]
  list_idx = list(np.argmax(list_idx[0, 2:], axis=1))  
  print(list_idx)
  for i in range(len(list_idx)-1):
    if list_idx[i]== text[-1] :
        continue
    text.append(list_idx[i])
  return text


def decode(list_idx,vocab=i2c):
  text=remove_dup(list_idx)
  text=[x for x in text if x !=0 and x!=87]
  text=[vocab[idx] for idx in text]
  return ''.join(text)


if __name__ == '__main__':

	model = get_Model(img_w, img_h,is_training=False)

	try:
	    model.load_weights('model.hdf5')
	    print("...Previous weight data...")
	except:
	    raise Exception("No weight file!")


	test_imgs = glob.glob('test_images/*.png')

	for test_img in test_imgs:
	    img = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)

	    img_pred = img.astype(np.float32)
	    img_pred = cv2.resize(img_pred, (img_w, img_h))
	    img_pred = (img_pred / 255.0) * 2.0 - 1.0
	    img_pred = img_pred.T
	    img_pred = np.expand_dims(img_pred, axis=-1)
	    img_pred = np.expand_dims(img_pred, axis=0)

	    net_out_value = model.predict(img_pred)

	    pred_texts = decode(net_out_value)
	    print(pred_texts)