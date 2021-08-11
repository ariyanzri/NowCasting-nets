from numpy.core.defchararray import index
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization
from tensorflow.keras.layers import Conv3D, MaxPool3D, Conv3DTranspose, Add
from tensorflow.keras.layers import SpatialDropout3D, UpSampling3D, Dropout, RepeatVector, Average
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mse, mae, Huber
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
mpl.rcParams['figure.dpi'] = 300
import argparse
import json
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import tensorflow.keras.layers as layers
import pickle

class UNET_Like_3D_Class:

    def __init__(self, dic):

        input_layer = Input(dic['input_shape'])
        
        if dic['input_bn']:
            x_init = BatchNormalization()(input_layer)
        else:
            x_init = input_layer
            
        x_conv1_b1 = Conv3D(dic['start_filter'], [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_init)
        x_conv2_b1 = Conv3D(dic['start_filter'], [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_conv1_b1)
        x_max_b1 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b1)
        x_bn_b1 = BatchNormalization()(x_max_b1)
        x_do_b1 = Dropout(dic['dr_rate'])(x_bn_b1)

        x_conv1_b2 = Conv3D(dic['start_filter']*2, [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_do_b1)
        x_conv2_b2 = Conv3D(dic['start_filter']*2, [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_conv1_b2)
        x_max_b2 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b2)
        x_bn_b2 = BatchNormalization()(x_max_b2)
        x_do_b2 = Dropout(dic['dr_rate'])(x_bn_b2)

        x_conv1_b3 = Conv3D(dic['start_filter']*4, [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_do_b2)
        x_conv2_b3 = Conv3D(dic['start_filter']*4, [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_conv1_b3)
        x_max_b3 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b3)
        x_bn_b3 = BatchNormalization()(x_max_b3)
        x_do_b3 = Dropout(dic['dr_rate'])(x_bn_b3)

        x_conv1_b4 = Conv3D(dic['start_filter']*8, [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_do_b3)
        x_conv2_b4 = Conv3D(dic['start_filter']*8, [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_conv1_b4)
        x_max_b4 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b4)
        x_bn_b4 = BatchNormalization()(x_max_b4)
        x_do_b4 = Dropout(dic['dr_rate'])(x_bn_b4)

        # ------- Head Normal Output (normal decoder)

        x_conv1_b5 = Conv3D(dic['start_filter']*8, [3, 1, 1], activation=dic['activation'])(x_do_b4)
        x_conv2_b5 = Conv3D(dic['start_filter']*8, [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b5)
        x_deconv_b5 = Conv3DTranspose(dic['start_filter']*8, [1, dic['deconv_kernel_size'], dic['deconv_kernel_size']],(1,2,2),padding='same', activation=dic['activation'])(x_conv2_b5)
        x_bn_b5 = BatchNormalization()(x_deconv_b5)
        x_do_b5 = Dropout(dic['dr_rate'])(x_bn_b5)

        cropped_x_conv2_b4 = layers.Cropping3D(cropping=((2,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b4)
        x_conv1_b6 = Conv3D(dic['start_filter']*4, [3, 1, 1], activation=dic['activation'])(layers.concatenate([cropped_x_conv2_b4,x_do_b5]))
        x_conv2_b6 = Conv3D(dic['start_filter']*4, [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b6)
        x_deconv_b6 = Conv3DTranspose(dic['start_filter']*4, [1, dic['deconv_kernel_size'], dic['deconv_kernel_size']],(1,2,2),padding='same', activation=dic['activation'])(x_conv2_b6)
        x_bn_b6 = BatchNormalization()(x_deconv_b6)
        x_do_b6 = Dropout(dic['dr_rate'])(x_bn_b6)

        cropped_x_conv2_b3 = layers.Cropping3D(cropping=((4,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b3)
        x_conv1_b7 = Conv3D(dic['start_filter']*2, [3, 1, 1], activation=dic['activation'])(layers.concatenate([cropped_x_conv2_b3,x_do_b6]))
        x_conv2_b7 = Conv3D(dic['start_filter']*2, [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b7)
        x_deconv_b7 = Conv3DTranspose(dic['start_filter']*2, [1, dic['deconv_kernel_size'], dic['deconv_kernel_size']],(1,2,2),padding='same', activation=dic['activation'])(x_conv2_b7)
        x_bn_b7 = BatchNormalization()(x_deconv_b7)
        x_do_b7 = Dropout(dic['dr_rate'])(x_bn_b7)

        cropped_x_conv2_b2 = layers.Cropping3D(cropping=((6,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b2)
        x_conv1_b8 = Conv3D(dic['start_filter'], [1, 1, 1],padding='same', activation=dic['activation'])(layers.concatenate([cropped_x_conv2_b2,x_do_b7]))
        x_conv2_b8 = Conv3D(dic['start_filter'], [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b8)
        x_deconv_b8 = Conv3DTranspose(dic['start_filter'], [1, dic['deconv_kernel_size'], dic['deconv_kernel_size']],(1,2,2),padding='same', activation=dic['activation'])(x_conv2_b8)
        x_bn_b8 = BatchNormalization()(x_deconv_b8)
        x_do_b8 = Dropout(dic['dr_rate'])(x_bn_b8)

        cropped_x_conv2_b1 = layers.Cropping3D(cropping=((6,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b1)
        x_conv1_b9 = Conv3D(int(dic['start_filter']/2), [1, 1, 1],padding='same', activation=dic['activation'])(layers.concatenate([cropped_x_conv2_b1,x_do_b8]))
        x_conv2_b9 = Conv3D(int(dic['start_filter']/2), [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b9)
        x_bn_b9 = BatchNormalization()(x_conv2_b9)
        x_do_b9 = Dropout(dic['dr_rate'])(x_bn_b9)

        normal_output = Conv3DTranspose(1, [1, 1, 1], activation='linear')(x_do_b9)

        # ----------

        model = Model(inputs=[input_layer], outputs=[normal_output])
        
        opt = eval(dic['optimizer'])

        if dic['loss'] == 'huber':
            loss = Huber()
        else:
            loss = dic['loss']

        model.compile(optimizer=opt(dic['lr']), loss=loss,
                      metrics=['mse', 'mae'])
        
        self.model = model

class UNET_3D_Residual_Class:

    def __init__(self, dic):

        input_layer = Input(dic['input_shape'])
        
        if dic['input_bn']:
            x_init = BatchNormalization()(input_layer)
        else:
            x_init = input_layer
            
        x_conv1_b1 = Conv3D(dic['start_filter'], [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_init)
        x_conv2_b1 = Conv3D(dic['start_filter'], [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_conv1_b1)
        x_max_b1 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b1)
        x_bn_b1 = BatchNormalization()(x_max_b1)
        x_do_b1 = Dropout(dic['dr_rate'])(x_bn_b1)

        x_conv1_b2 = Conv3D(dic['start_filter']*2, [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_do_b1)
        x_conv2_b2 = Conv3D(dic['start_filter']*2, [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_conv1_b2)
        x_max_b2 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b2)
        x_bn_b2 = BatchNormalization()(x_max_b2)
        x_do_b2 = Dropout(dic['dr_rate'])(x_bn_b2)

        x_conv1_b3 = Conv3D(dic['start_filter']*4, [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_do_b2)
        x_conv2_b3 = Conv3D(dic['start_filter']*4, [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_conv1_b3)
        x_max_b3 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b3)
        x_bn_b3 = BatchNormalization()(x_max_b3)
        x_do_b3 = Dropout(dic['dr_rate'])(x_bn_b3)

        x_conv1_b4 = Conv3D(dic['start_filter']*8, [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_do_b3)
        x_conv2_b4 = Conv3D(dic['start_filter']*8, [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_conv1_b4)
        x_max_b4 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b4)
        x_bn_b4 = BatchNormalization()(x_max_b4)
        x_do_b4 = Dropout(dic['dr_rate'])(x_bn_b4)

        # ------- Head Residual Output (Residual Decoder)

        x_conv1_b5 = Conv3D(dic['start_filter']*8, [3, 1, 1], activation=dic['activation'])(x_do_b4)
        x_conv2_b5 = Conv3D(dic['start_filter']*8, [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b5)
        x_deconv_b5 = Conv3DTranspose(dic['start_filter']*8, [1, dic['deconv_kernel_size'], dic['deconv_kernel_size']],(1,2,2),padding='same', activation=dic['activation'])(x_conv2_b5)
        x_bn_b5 = BatchNormalization()(x_deconv_b5)
        x_do_b5 = Dropout(dic['dr_rate'])(x_bn_b5)

        cropped_x_conv2_b4 = layers.Cropping3D(cropping=((8,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b4)
        cropped_x_conv2_b4 = layers.concatenate([cropped_x_conv2_b4]*7,axis=1)
        x_conv1_b6 = Conv3D(dic['start_filter']*4, [3, 1, 1], activation=dic['activation'])(layers.concatenate([cropped_x_conv2_b4,x_do_b5]))
        x_conv2_b6 = Conv3D(dic['start_filter']*4, [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b6)
        x_deconv_b6 = Conv3DTranspose(dic['start_filter']*4, [1, dic['deconv_kernel_size'], dic['deconv_kernel_size']],(1,2,2),padding='same', activation=dic['activation'])(x_conv2_b6)
        x_bn_b6 = BatchNormalization()(x_deconv_b6)
        x_do_b6 = Dropout(dic['dr_rate'])(x_bn_b6)

        cropped_x_conv2_b3 = layers.Cropping3D(cropping=((8,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b3)
        cropped_x_conv2_b3 = layers.concatenate([cropped_x_conv2_b3]*5,axis=1)
        x_conv1_b7 = Conv3D(dic['start_filter']*2, [3, 1, 1], activation=dic['activation'])(layers.concatenate([cropped_x_conv2_b3,x_do_b6]))
        x_conv2_b7 = Conv3D(dic['start_filter']*2, [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b7)
        x_deconv_b7 = Conv3DTranspose(dic['start_filter']*2, [1, dic['deconv_kernel_size'], dic['deconv_kernel_size']],(1,2,2),padding='same', activation=dic['activation'])(x_conv2_b7)
        x_bn_b7 = BatchNormalization()(x_deconv_b7)
        x_do_b7 = Dropout(dic['dr_rate'])(x_bn_b7)

        cropped_x_conv2_b2 = layers.Cropping3D(cropping=((8,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b2)
        cropped_x_conv2_b2 = layers.concatenate([cropped_x_conv2_b2]*3,axis=1)
        x_conv1_b8 = Conv3D(dic['start_filter'], [1, 1, 1],padding='same', activation=dic['activation'])(layers.concatenate([cropped_x_conv2_b2,x_do_b7]))
        x_conv2_b8 = Conv3D(dic['start_filter'], [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b8)
        x_deconv_b8 = Conv3DTranspose(dic['start_filter'], [1, dic['deconv_kernel_size'], dic['deconv_kernel_size']],(1,2,2),padding='same', activation=dic['activation'])(x_conv2_b8)
        x_bn_b8 = BatchNormalization()(x_deconv_b8)
        x_do_b8 = Dropout(dic['dr_rate'])(x_bn_b8)

        cropped_x_conv2_b1 = layers.Cropping3D(cropping=((8,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b1)
        cropped_x_conv2_b1 = layers.concatenate([cropped_x_conv2_b1]*3,axis=1)
        x_conv1_b9 = Conv3D(int(dic['start_filter']/2), [1, 1, 1],padding='same', activation=dic['activation'])(layers.concatenate([cropped_x_conv2_b1,x_do_b8]))
        x_conv2_b9 = Conv3D(int(dic['start_filter']/2), [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b9)
        x_bn_b9 = BatchNormalization()(x_conv2_b9)
        x_do_b9 = Dropout(dic['dr_rate'])(x_bn_b9)
        
        residual_output = Conv3DTranspose(1, [1, 1, 1], activation='linear')(x_do_b9)

        last_timestep_input_residual = layers.Cropping3D(cropping=((8,0),(0,0),(0,0)),data_format="channels_last")(input_layer)
        last_timestep_input_residual = layers.concatenate([last_timestep_input_residual]*3,axis=1)
        residual_output = Add()([last_timestep_input_residual, residual_output])

        # ----------

        model = Model(inputs=[input_layer], outputs=[residual_output])
        
        opt = eval(dic['optimizer'])

        if dic['loss'] == 'huber':
            loss = Huber()
        else:
            loss = dic['loss']

        model.compile(optimizer=opt(dic['lr']), loss=loss,
                      metrics=['mse', 'mae'])
        
        self.model = model

class UNET_3D_Both_Class:

    def __init__(self, dic):

        input_layer = Input(dic['input_shape'])
        
        if dic['input_bn']:
            x_init = BatchNormalization()(input_layer)
        else:
            x_init = input_layer
            
        x_conv1_b1 = Conv3D(dic['start_filter'], [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_init)
        x_conv2_b1 = Conv3D(dic['start_filter'], [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_conv1_b1)
        x_max_b1 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b1)
        x_bn_b1 = BatchNormalization()(x_max_b1)
        x_do_b1 = Dropout(dic['dr_rate'])(x_bn_b1)

        x_conv1_b2 = Conv3D(dic['start_filter']*2, [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_do_b1)
        x_conv2_b2 = Conv3D(dic['start_filter']*2, [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_conv1_b2)
        x_max_b2 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b2)
        x_bn_b2 = BatchNormalization()(x_max_b2)
        x_do_b2 = Dropout(dic['dr_rate'])(x_bn_b2)

        x_conv1_b3 = Conv3D(dic['start_filter']*4, [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_do_b2)
        x_conv2_b3 = Conv3D(dic['start_filter']*4, [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_conv1_b3)
        x_max_b3 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b3)
        x_bn_b3 = BatchNormalization()(x_max_b3)
        x_do_b3 = Dropout(dic['dr_rate'])(x_bn_b3)

        x_conv1_b4 = Conv3D(dic['start_filter']*8, [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_do_b3)
        x_conv2_b4 = Conv3D(dic['start_filter']*8, [1, dic['conv_kernel_size'], dic['conv_kernel_size']],padding='same', activation=dic['activation'])(x_conv1_b4)
        x_max_b4 = MaxPool3D([1, 2, 2],padding='same')(x_conv2_b4)
        x_bn_b4 = BatchNormalization()(x_max_b4)
        x_do_b4 = Dropout(dic['dr_rate'])(x_bn_b4)

        # ------- Head Normal Output (normal decoder)

        x_conv1_b5 = Conv3D(dic['start_filter']*8, [3, 1, 1], activation=dic['activation'])(x_do_b4)
        x_conv2_b5 = Conv3D(dic['start_filter']*8, [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b5)
        x_deconv_b5 = Conv3DTranspose(dic['start_filter']*8, [1, dic['deconv_kernel_size'], dic['deconv_kernel_size']],(1,2,2),padding='same', activation=dic['activation'])(x_conv2_b5)
        x_bn_b5 = BatchNormalization()(x_deconv_b5)
        x_do_b5 = Dropout(dic['dr_rate'])(x_bn_b5)

        cropped_x_conv2_b4 = layers.Cropping3D(cropping=((2,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b4)
        x_conv1_b6 = Conv3D(dic['start_filter']*4, [3, 1, 1], activation=dic['activation'])(layers.concatenate([cropped_x_conv2_b4,x_do_b5]))
        x_conv2_b6 = Conv3D(dic['start_filter']*4, [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b6)
        x_deconv_b6 = Conv3DTranspose(dic['start_filter']*4, [1, dic['deconv_kernel_size'], dic['deconv_kernel_size']],(1,2,2),padding='same', activation=dic['activation'])(x_conv2_b6)
        x_bn_b6 = BatchNormalization()(x_deconv_b6)
        x_do_b6 = Dropout(dic['dr_rate'])(x_bn_b6)

        cropped_x_conv2_b3 = layers.Cropping3D(cropping=((4,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b3)
        x_conv1_b7 = Conv3D(dic['start_filter']*2, [3, 1, 1], activation=dic['activation'])(layers.concatenate([cropped_x_conv2_b3,x_do_b6]))
        x_conv2_b7 = Conv3D(dic['start_filter']*2, [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b7)
        x_deconv_b7 = Conv3DTranspose(dic['start_filter']*2, [1, dic['deconv_kernel_size'], dic['deconv_kernel_size']],(1,2,2),padding='same', activation=dic['activation'])(x_conv2_b7)
        x_bn_b7 = BatchNormalization()(x_deconv_b7)
        x_do_b7 = Dropout(dic['dr_rate'])(x_bn_b7)

        cropped_x_conv2_b2 = layers.Cropping3D(cropping=((6,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b2)
        x_conv1_b8 = Conv3D(dic['start_filter'], [1, 1, 1],padding='same', activation=dic['activation'])(layers.concatenate([cropped_x_conv2_b2,x_do_b7]))
        x_conv2_b8 = Conv3D(dic['start_filter'], [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b8)
        x_deconv_b8 = Conv3DTranspose(dic['start_filter'], [1, dic['deconv_kernel_size'], dic['deconv_kernel_size']],(1,2,2),padding='same', activation=dic['activation'])(x_conv2_b8)
        x_bn_b8 = BatchNormalization()(x_deconv_b8)
        x_do_b8 = Dropout(dic['dr_rate'])(x_bn_b8)

        cropped_x_conv2_b1 = layers.Cropping3D(cropping=((6,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b1)
        x_conv1_b9 = Conv3D(int(dic['start_filter']/2), [1, 1, 1],padding='same', activation=dic['activation'])(layers.concatenate([cropped_x_conv2_b1,x_do_b8]))
        x_conv2_b9 = Conv3D(int(dic['start_filter']/2), [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b9)
        x_bn_b9 = BatchNormalization()(x_conv2_b9)
        x_do_b9 = Dropout(dic['dr_rate'])(x_bn_b9)

        normal_output = Conv3DTranspose(1, [1, 1, 1], activation='linear')(x_do_b9)

        # ------- Head Residual Output (Residual Decoder)

        x_conv1_b5 = Conv3D(dic['start_filter']*8, [3, 1, 1], activation=dic['activation'])(x_max_b4)
        x_conv2_b5 = Conv3D(dic['start_filter']*8, [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b5)
        x_deconv_b5 = Conv3DTranspose(dic['start_filter']*8, [1, dic['deconv_kernel_size'], dic['deconv_kernel_size']],(1,2,2),padding='same', activation=dic['activation'])(x_conv2_b5)
        x_bn_b5 = BatchNormalization()(x_deconv_b5)
        x_do_b5 = Dropout(dic['dr_rate'])(x_bn_b5)

        cropped_x_conv2_b4 = layers.Cropping3D(cropping=((8,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b4)
        cropped_x_conv2_b4 = layers.concatenate([cropped_x_conv2_b4]*7,axis=1)
        x_conv1_b6 = Conv3D(dic['start_filter']*4, [3, 1, 1], activation=dic['activation'])(layers.concatenate([cropped_x_conv2_b4,x_do_b5]))
        x_conv2_b6 = Conv3D(dic['start_filter']*4, [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b6)
        x_deconv_b6 = Conv3DTranspose(dic['start_filter']*4, [1, dic['deconv_kernel_size'], dic['deconv_kernel_size']],(1,2,2),padding='same', activation=dic['activation'])(x_conv2_b6)
        x_bn_b6 = BatchNormalization()(x_deconv_b6)
        x_do_b6 = Dropout(dic['dr_rate'])(x_bn_b6)

        cropped_x_conv2_b3 = layers.Cropping3D(cropping=((8,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b3)
        cropped_x_conv2_b3 = layers.concatenate([cropped_x_conv2_b3]*5,axis=1)
        x_conv1_b7 = Conv3D(dic['start_filter']*2, [3, 1, 1], activation=dic['activation'])(layers.concatenate([cropped_x_conv2_b3,x_do_b6]))
        x_conv2_b7 = Conv3D(dic['start_filter']*2, [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b7)
        x_deconv_b7 = Conv3DTranspose(dic['start_filter']*2, [1, dic['deconv_kernel_size'], dic['deconv_kernel_size']],(1,2,2),padding='same', activation=dic['activation'])(x_conv2_b7)
        x_bn_b7 = BatchNormalization()(x_deconv_b7)
        x_do_b7 = Dropout(dic['dr_rate'])(x_bn_b7)

        cropped_x_conv2_b2 = layers.Cropping3D(cropping=((8,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b2)
        cropped_x_conv2_b2 = layers.concatenate([cropped_x_conv2_b2]*3,axis=1)
        x_conv1_b8 = Conv3D(dic['start_filter'], [1, 1, 1],padding='same', activation=dic['activation'])(layers.concatenate([cropped_x_conv2_b2,x_do_b7]))
        x_conv2_b8 = Conv3D(dic['start_filter'], [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b8)
        x_deconv_b8 = Conv3DTranspose(dic['start_filter'], [1, dic['deconv_kernel_size'], dic['deconv_kernel_size']],(1,2,2),padding='same', activation=dic['activation'])(x_conv2_b8)
        x_bn_b8 = BatchNormalization()(x_deconv_b8)
        x_do_b8 = Dropout(dic['dr_rate'])(x_bn_b8)

        cropped_x_conv2_b1 = layers.Cropping3D(cropping=((8,0),(0,0),(0,0)),data_format="channels_last")(x_conv2_b1)
        cropped_x_conv2_b1 = layers.concatenate([cropped_x_conv2_b1]*3,axis=1)

        x_conv1_b9 = Conv3D(int(dic['start_filter']/2), [1, 1, 1],padding='same', activation=dic['activation'])(layers.concatenate([cropped_x_conv2_b1,x_do_b8]))
        x_conv2_b9 = Conv3D(int(dic['start_filter']/2), [1, 1, 1],padding='same', activation=dic['activation'])(x_conv1_b9)
        x_bn_b9 = BatchNormalization()(x_conv2_b9)
        x_do_b9 = Dropout(dic['dr_rate'])(x_bn_b9)

        residual_output = Conv3DTranspose(1, [1, 1, 1], activation='linear')(x_do_b9)
        last_timestep_input_residual = layers.Cropping3D(cropping=((8,0),(0,0),(0,0)),data_format="channels_last")(input_layer)
        last_timestep_input_residual = layers.concatenate([last_timestep_input_residual]*3,axis=1)
        residual_output = Add()([last_timestep_input_residual, residual_output])

        # ---------- Averaging the two output

        output = Average()([normal_output,residual_output])


        model = Model(inputs=[input_layer], outputs=[output])
        
        opt = eval(dic['optimizer'])

        if dic['loss'] == 'huber':
            loss = Huber()
        else:
            loss = dic['loss']

        model.compile(optimizer=opt(dic['lr']), loss=loss,
                      metrics=['mse', 'mae'])
        
        self.model = model

class CONV_LSTM_Class:

    def __init__(self, dic):

        input_layer = Input(dic['input_shape'])
        x = input_layer
        
        x = ConvLSTM2D(filters=dic['start_filter'], kernel_size=(dic['conv_kernel_size'], dic['conv_kernel_size']), padding="same", activation='relu', return_sequences=True)(x)
        x = Conv3D(dic['start_filter'], [3, 1, 1], activation='relu')(x)
        
        x = ConvLSTM2D(filters=int(dic['start_filter']/2), kernel_size=(dic['conv_kernel_size'], dic['conv_kernel_size']), padding="same", activation='relu', return_sequences=True)(x)
        x = Conv3D(int(dic['start_filter']/2), [3, 1, 1], activation='relu')(x)
        
        x = ConvLSTM2D(filters=int(dic['start_filter']/4), kernel_size=(dic['conv_kernel_size'], dic['conv_kernel_size']), padding="same", activation='relu', return_sequences=True)(x)
        x = Conv3D(int(dic['start_filter']/4), [3, 1, 1], activation='relu')(x)
        
        x = Conv3D(1, [1, 1, 1], padding='same', activation='relu')(x)    
        
        model = Model(inputs=[input_layer], outputs=[x])
        
        opt = eval(dic['optimizer'])
        
        if dic['loss'] == 'huber':
            loss = Huber()
        else:
            loss = dic['loss']

        model.compile(optimizer=opt(dic['lr']), loss=loss,
                      metrics=['mse', 'mae'])
        #
        self.model = model

class CONV_LSTM_Residual_Class:

    def __init__(self, dic):

        input_layer = Input(dic['input_shape'])
        x = input_layer

        x = ConvLSTM2D(filters=dic['start_filter'], kernel_size=(dic['conv_kernel_size'], dic['conv_kernel_size']), padding="same", activation='relu', return_sequences=True)(x)
        x = Conv3D(dic['start_filter'], [3, 1, 1], activation='relu')(x)
        
        x = ConvLSTM2D(filters=int(dic['start_filter']/2), kernel_size=(dic['conv_kernel_size'], dic['conv_kernel_size']), padding="same", activation='relu', return_sequences=True)(x)
        x = Conv3D(int(dic['start_filter']/2), [3, 1, 1], activation='relu')(x)
        
        x = ConvLSTM2D(filters=int(dic['start_filter']/4), kernel_size=(dic['conv_kernel_size'], dic['conv_kernel_size']), padding="same", activation='relu', return_sequences=True)(x)
        x = Conv3D(int(dic['start_filter']/4), [3, 1, 1], activation='relu')(x)
        
        x = Conv3D(1, [1, 1, 1], padding='same', activation='relu')(x)    
        
        last_ts = layers.Cropping3D(cropping=((8,0),(0,0),(0,0)),data_format="channels_last")(input_layer)
        output = Add()([last_ts, x])

        model = Model(inputs=[input_layer], outputs=[output])
        
        opt = eval(dic['optimizer'])
        
        if dic['loss'] == 'huber':
            loss = Huber()
        else:
            loss = dic['loss']

        model.compile(optimizer=opt(dic['lr']), loss=loss,
                      metrics=['mse', 'mae'])
        #
        self.model = model

class CONV_LSTM_Both_Class:

    def __init__(self, dic):

        input_layer = Input(dic['input_shape'])

        # Normal Model

        x_normal = input_layer

        x_normal = ConvLSTM2D(filters=dic['start_filter'], kernel_size=(dic['conv_kernel_size'], dic['conv_kernel_size']), padding="same", activation='relu', return_sequences=True)(x_normal)
        x_normal = Conv3D(dic['start_filter'], [3, 1, 1], activation='relu')(x_normal)
        
        x_normal = ConvLSTM2D(filters=int(dic['start_filter']/2), kernel_size=(dic['conv_kernel_size'], dic['conv_kernel_size']), padding="same", activation='relu', return_sequences=True)(x_normal)
        x_normal = Conv3D(int(dic['start_filter']/2), [3, 1, 1], activation='relu')(x_normal)
        
        x_normal = ConvLSTM2D(filters=int(dic['start_filter']/4), kernel_size=(dic['conv_kernel_size'], dic['conv_kernel_size']), padding="same", activation='relu', return_sequences=True)(x_normal)
        x_normal = Conv3D(int(dic['start_filter']/4), [3, 1, 1], activation='relu')(x_normal)
        
        output_normal = Conv3D(1, [1, 1, 1], padding='same', activation='relu')(x_normal)
        
        # Residual Model

        x_residual = input_layer
        x_residual = ConvLSTM2D(filters=dic['start_filter'], kernel_size=(dic['conv_kernel_size'], dic['conv_kernel_size']), padding="same", activation='relu', return_sequences=True)(x_residual)
        x_residual = Conv3D(dic['start_filter'], [3, 1, 1], activation='relu')(x_residual)
        
        x_residual = ConvLSTM2D(filters=int(dic['start_filter']/2), kernel_size=(dic['conv_kernel_size'], dic['conv_kernel_size']), padding="same", activation='relu', return_sequences=True)(x_residual)
        x_residual = Conv3D(int(dic['start_filter']/2), [3, 1, 1], activation='relu')(x_residual)
        
        x_residual = ConvLSTM2D(filters=int(dic['start_filter']/4), kernel_size=(dic['conv_kernel_size'], dic['conv_kernel_size']), padding="same", activation='relu', return_sequences=True)(x_residual)
        x_residual = Conv3D(int(dic['start_filter']/4), [3, 1, 1], activation='relu')(x_residual)
        
        x_residual = Conv3D(1, [1, 1, 1], padding='same', activation='relu')(x_residual)
        
        last_ts = layers.Cropping3D(cropping=((8,0),(0,0),(0,0)),data_format="channels_last")(input_layer)
        output_residual = Add()([last_ts, x_residual])

        # Model 

        output = Average()([output_normal,output_residual])

        model = Model(inputs=[input_layer], outputs=[output])
        
        opt = eval(dic['optimizer'])
        
        if dic['loss'] == 'huber':
            loss = Huber()
        else:
            loss = dic['loss']

        model.compile(optimizer=opt(dic['lr']), loss=loss,
                      metrics=['mse', 'mae'])
        #
        self.model = model

class LinearRegression_Class:

    def load_model(path):
        mdl = LinearRegression_Class()
        with open(path,'rb') as f:
            mdl.sk_model = pickle.load(f)

        return mdl

    def reshape(arr):
        arr = arr.squeeze()
        arr = arr.reshape(arr.shape[0],arr.shape[1]*arr.shape[2])
        arr = arr.swapaxes(0,1)
        return arr

    def reshape_back(arr,s):
        arr = arr.swapaxes(0,1)
        arr = arr.reshape(s[0],s[1],s[2])
        return arr

    def fit(self, X_train, y_train):
        self.sk_model = LinearRegression().fit(X_train, y_train)

    def predict(self,X):
        s = X.squeeze().shape
        
        reshaped_X = LinearRegression_Class.reshape(X)
        reshaped_Y = self.sk_model.predict(reshaped_X)
        Y = LinearRegression_Class.reshape_back(reshaped_Y,(3,s[1],s[2]))

        return Y
        
class RandomForrest_Class:

    def __init__(self):
        self.sk_model = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=20)

    def load_model(path):
        mdl = RandomForrest_Class()
        with open(path,'rb') as f:
            mdl.sk_model = pickle.load(f)

        return mdl

    def reshape(arr):
        arr = arr.squeeze()
        arr = arr.reshape(arr.shape[0],arr.shape[1]*arr.shape[2])
        arr = arr.swapaxes(0,1)
        return arr

    def reshape_back(arr,s):
        arr = arr.swapaxes(0,1)
        arr = arr.reshape(s[0],s[1],s[2])
        return arr

    def get_sample_indexes(arr):
        indexes = np.random.choice(np.array(range(arr.shape[0])),10000000)
        return indexes

    def fit(self, X_train, y_train):
        indexes = RandomForrest_Class.get_sample_indexes(X_train)

        X = X_train[indexes]
        y = y_train[indexes]

        self.sk_model = self.sk_model.fit(X,y)

    def predict(self,X):
        s = X.squeeze().shape
        
        reshaped_X = RandomForrest_Class.reshape(X)
        reshaped_Y = self.sk_model.predict(reshaped_X)
        Y = RandomForrest_Class.reshape_back(reshaped_Y,(3,s[1],s[2]))

        return Y
