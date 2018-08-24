
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

train_dir = '00_input/train'
im_size = 100
classes = 50

def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            filename, class_id = line.rstrip('\n').split(',')
            res[filename] = int(class_id)
    return res


from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence

from keras.layers import (Input, concatenate, Conv2D, MaxPooling2D, 
                          UpSampling2D, Convolution2D, ZeroPadding2D, 
                          BatchNormalization, Activation, concatenate, 
                          Flatten, Dense, merge, Dropout)
from keras.optimizers import Adam
from keras.applications import vgg16


def get_model():
    model = vgg16.VGG16(
        weights='imagenet', 
        include_top=False, 
        input_shape=(im_size, im_size, 3))
    
    flatten = Flatten()(model.output)
    dense = Dense(classes, activation='softmax')(flatten)
    
    inputs = model.input
    model = Model(inputs=inputs, outputs=dense)
    model.compile(
        optimizer=Adam(0.0005, decay=0.00001), 
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


def train_detector(
        train_gt, 
        train_img_dir, 
        fast_train=False, 
        model_func=None, 
        model_name='{epoch:d}_{val_loss:.4f}.hdf5'):
    
    def parse(train_gt, train_img_dir, info=False, fast_train=False):
        from skimage.data import imread
        from scipy.ndimage import zoom
        from sklearn.utils import shuffle
        if fast_train:
            train_X = np.zeros((500, im_size, im_size, 3))
            train_Y = np.zeros((500, classes))
        else:
            train_X = np.zeros((len(train_gt), im_size, im_size, 3))
            train_Y = np.zeros((len(train_gt), classes))
        for i, img_name in enumerate(train_gt):
            if i == 500 and fast_train:
                break
            img = imread(train_img_dir+'/'+img_name, as_grey=False)
            train_Y[i][train_gt[img_name]] = 1
            img = zoom(img, [im_size/img.shape[0], im_size/img.shape[1], 1])
            img = (img / 255)
            train_X[i,:,:,:] = img
            del(img)
            if info and (i+1)%100 == 0:
                print('Image: ', i+1, end='\r')
        train_X, train_Y = shuffle(train_X, train_Y)
        return train_X, train_Y
    
    train_X, train_Y = parse(train_gt, train_img_dir, True, fast_train)
    if model_func == None:
        model = get_model()
    else:
        model = model_func()
        model_name += '_{epoch:d}_{val_loss:.4f}.hdf5'
    model.summary()
    checkpoint = ModelCheckpoint(
        model_name, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        period=1,
        save_weights_only=False)
    if fast_train:
        epochs = 1
        model.fit(train_X, train_Y, epochs=epochs, batch_size=40)
    else:
        epochs = 100
        try:
            model.fit(train_X, train_Y, epochs=epochs, batch_size=150, callbacks=[checkpoint], validation_split=(1/6))
        except KeyboardInterrupt:
            print('\nTraining interrupted')
    return model


def detect(model, test_img_dir):
    from os import listdir
    from skimage.data import imread
    from scipy.ndimage import zoom
    img_list = listdir(test_img_dir)
    data = np.zeros((100, im_size, im_size, 3))
    sizes = []
    k = 0
    ans = {}
    for i, img_name in enumerate(img_list):
        img = imread(test_img_dir+'/'+img_name, as_grey=False)
        sizes.append([img_name, img.shape])
        img = zoom(img, [im_size/img.shape[0], im_size/img.shape[1], 1])
        img = (img / 255)
        data[k,:,:,:] = img
        k += 1
        del(img)
        if (i+1)%100 == 0:
            print((i+1), ' images')
            points = model.predict(data, verbose=1)
            for q in range(len(points)):
                for j in range(classes):
                    if points[q][j] != 0:
                        ans[sizes[q][0]] = j
            sizes = []
            k = 0
            data = np.zeros((100, im_size, im_size, 1))
    if k != 0:
        data = data[:k,:,:,:]
        points = model.predict(data, verbose=1)
        for q in range(len(points)):
            for j in range(classes):
                if points[q][j] != 0:
                    ans[sizes[q][0]] = j
    return ans