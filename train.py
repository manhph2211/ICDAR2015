from model import *
from dataset import Generator
from keras import backend as K
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
from config import *

K.set_learning_phase(0)

# # Model description and training
model = get_Model(img_w, img_h,is_training=True)

try:
    model.load_weights('model.hdf5')
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass

train_G = Generator(train_label_file, img_w, img_h, batch_size, downsample_factor)
val_G = Generator(test_label_file, img_w, img_h, batch_size, downsample_factor)

optimizer = Adadelta(learning_rate=0.001)

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)
checkpoint = ModelCheckpoint(filepath='model.hdf5', monitor='loss', verbose=1, mode='min', period=1)

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

model.fit_generator(generator=train_G.next_batch(),
                    steps_per_epoch=int(train_G.n / batch_size),
                    epochs=30,
                    callbacks=[checkpoint],
                    validation_data=val_G.next_batch(),
                    validation_steps=int(val_G.n /batch_size))