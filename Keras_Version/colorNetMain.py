from __future__ import print_function
import argparse
import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input 
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate, Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Keras ColorNet')


parser.add_argument('path',type=str, help='path to the dataset (train or predict)\ncategory folders are needed')
parser.add_argument('action', metavar='action', type=str, help='train or predict')


parser.add_argument('--valpath', metavar='validate_DIR', default='', help='path to the validation dataset')
parser.add_argument('--epoch', default=500, type=int, metavar= 'N',
        help='number of total epochs to run')


parser.add_argument('--patience', default=-1, type=int, metavar= 'N',
        help='stop the epoch earlier when validatation loss doesn\'t drop')


parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')




inception = InceptionResNetV2(weights=None, include_top=True)
inception.load_weights('inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
inception.graph = tf.get_default_graph()

batch_size=20



def makeModel():
    embed_input = Input(shape=(1000,))
    encoder_input = Input(shape=(256, 256, 1,))
    encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

    fusion_output = RepeatVector(32 * 32)(embed_input) 
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output) 

    
    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)

    model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
    return model



def creat_inception_embedding(grayscaled_rgb):
        grayscaled_rgb_resized = []
        for i in grayscaled_rgb:
            i = resize(i, (299, 299, 3), mode='constant')
            grayscaled_rgb_resized.append(i)
        grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
        #grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
        with inception.graph.as_default():
            embed = inception.predict(grayscaled_rgb_resized)
        return embed


def image_gen(datagen):
    for batch in datagen:
        if batch[0].max() > 1:
            batch = batch/255.0

        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        #embed = create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([X_batch, creat_inception_embedding(grayscaled_rgb)], Y_batch)




def main():

    global args, best_loss
    args = parser.parse_args()

    if args.action == 'train':
        generator_t = (ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=20,
            horizontal_flip=True).flow_from_directory(directory=args.path,
                    class_mode=None, batch_size=batch_size, shuffle=True))
        data_gen = image_gen(generator_t)

    elif args.action == 'predict':
        generator_p = ImageDataGenerator(
            ).flow_from_directory(directory=args.path,
                class_mode=None, batch_size=batch_size)
        data_gen_p = image_gen(generator_p)
    else:
        raise Exception('the \'action\' argument need to be train or predict!')


    
    if args.valpath:
        generator_v = ImageDataGenerator(
            ).flow_from_directory(directory=args.valpath,
                class_mode=None, batch_size=batch_size, shuffle=True)
        data_gen_val = image_gen(generator_v)



    if args.action == 'predict':
        pass


    else:

        model = makeModel()

        if args.resume:
            model.load_weights(args.resume)

        callback = [ModelCheckpoint('weight.best.hdf5', verbose=0, save_best_only=True)]
        if args.patience != -1:
            callback.extend(keras.callbacks.EarlyStopping(monitor='val_loss', 
                patience=args.patience, verbose=1, mode='auto'))



        model.compile(optimizer='adam', loss='mse')

        if not args.valpath:
            model.fit_generator(data_gen, epochs=args.epoch, steps_per_epoch= min(len(generator_t),500),
                callbacks=callback)
        else:
            model.fit_generator(data_gen, epochs=args.epoch, steps_per_epoch= min(len(generator_v),50),
                validation_data=data_gen_val,validation_steps=len(datagen_val),
                callbacks=callback)







if __name__ == '__main__':
    main()





            




    










