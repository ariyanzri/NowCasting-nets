from tensorflow import keras
import tensorflow as tf
import h5py
import numpy as np
import argparse
import json
import pickle

from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization
from tensorflow.keras.layers import Conv3D, MaxPool3D, Conv3DTranspose
from tensorflow.keras.layers import SpatialDropout3D, UpSampling3D, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mse, mae, Huber
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import Models
from get_data import train_test, train_test_residuals, train_test_experiment1

def get_args():
    parser = argparse.ArgumentParser(
        description='IMERG_test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p',
                        '--path',
                        help='The path to the directory that contains the experimetns json files',
                        metavar='path',
                        required=True)

    parser.add_argument('-e',
                        '--exp',
                        help='The name of the experiment',
                        metavar='exp',
                        required=True)

    parser.add_argument('-c',
                        '--config',
                        help='The path to the config file',
                        metavar='config',
                        required=True)


    return parser.parse_args()

def create_model_base_line(file,experiment_name,dic):

    def reshape(arr):
        arr = arr.squeeze()
        arr = arr.reshape(arr.shape[0],arr.shape[1],arr.shape[2]*arr.shape[3])
        arr = arr.swapaxes(0,1)
        arr = arr.reshape(arr.shape[0],arr.shape[1]*arr.shape[2]).T

        return arr

    X_train, X_valid, y_train, y_valid = train_test(file)

    X_valid_old = X_valid.copy()

    X_train = reshape(X_train)
    y_train = reshape(y_train)
    X_valid = reshape(X_valid)
    y_valid = reshape(y_valid)
    
    if experiment_name == 'LREG':
        mdl = Models.LinearRegression_Class()
        mdl.fit(X_train, y_train)
        print("Linear regressor fitted.")

        pickle.dump(mdl.sk_model, open('{}/{}.sav'.format(dic['directories']['models'], experiment_name), 'wb'))

        print(mdl.sk_model.score(X_train,y_train))
        print(mdl.sk_model.score(X_valid,y_valid))
        
    elif experiment_name == 'RFOR':
        mdl = Models.RandomForrest_Class()
        mdl.fit(X_train, y_train)
        print("Random forrest regressor fitted.")

        pickle.dump(mdl.sk_model, open('{}/{}.sav'.format(dic['directories']['models'], experiment_name), 'wb'))

        print(mdl.sk_model.score(X_train[2000:2500],y_train[2000:2500]))
        print(mdl.sk_model.score(X_valid[2000:2500],y_valid[2000:2500]))

def create_model(file, experiment_name, dic):
    
    X_train, X_valid, y_train, y_valid = train_test(file)

    dic['experiments'][experiment_name]['input_shape'] = X_train.shape[1:]
    
    if experiment_name.split('_')[0] == 'CONVLSTM':
        model = Models.CONV_LSTM_Class(dic['experiments'][experiment_name]).model
    
    elif experiment_name.split('_')[0] == 'CONVLSTMR':
        model = Models.CONV_LSTM_Residual_Class(dic['experiments'][experiment_name]).model
    
    elif experiment_name.split('_')[0] == 'CONVLSTMB':
        model = Models.CONV_LSTM_Both_Class(dic['experiments'][experiment_name]).model
    
    elif experiment_name.split('_')[0] == 'UNET':
        model = Models.UNET_Like_3D_Class(dic['experiments'][experiment_name]).model

    elif experiment_name.split('_')[0] == 'UNETR':
        model = Models.UNET_3D_Residual_Class(dic['experiments'][experiment_name]).model

    elif experiment_name.split('_')[0] == 'UNETB':
        model = Models.UNET_3D_Both_Class(dic['experiments'][experiment_name]).model
    
    elif experiment_name.split('_')[0] == 'CONV3D':
        model = Models.CONV_3D_Simple_Class(dic['experiments'][experiment_name]).model

    early_stopping_clbk = EarlyStopping(monitor='val_loss', min_delta=0, \
        patience=dic['experiments'][experiment_name]['patience'], verbose=0,mode='auto', baseline=None, restore_best_weights=True)
 
    history = model.fit(X_train, y_train, epochs=dic['experiments'][experiment_name]['epochs'], batch_size=dic['experiments'][experiment_name]['batch_size'], 
                        validation_data=(X_valid, y_valid), verbose=2,callbacks=[early_stopping_clbk])


    exp_dict = {'settings':dic['experiments'][experiment_name],'results':{}}
    exp_dict['results'] = history.history
    
    new_results = {}

    for key in exp_dict['results']:
        tmp = []
        for a in exp_dict['results'][key]:
            tmp.append(float(str(a)))
        new_results[key] = tmp

    exp_dict['results'] = new_results

    with open('{0}/results_{1}.json'.format(dic['directories']['results'],experiment_name), 'w') as file_pi:
        json.dump(exp_dict,file_pi)
    
    model.save('{}/{}.h5'.format(dic['directories']['models'], experiment_name))

    return model, history

def main():
    args = get_args() 
    experiment_name = args.exp
    path_setting = args.path
    config_file = args.config

    dic = {}
    with open(config_file) as f:
        dic["directories"] = json.load(f)

    ds_file = dic['directories']['datasets']+'/final_dataset_IMERG_Jun_14.h5'
 
    if experiment_name == 'LREG' or experiment_name == 'RFOR':
        create_model_base_line(ds_file,experiment_name,dic)
    else:
        with open("{0}/experiments_{1}.json".format(path_setting,experiment_name.split('_')[0])) as f:
            dic["experiments"] = json.load(f)

        model, history = create_model(ds_file, experiment_name, dic)

if __name__ == "__main__":
    main()
