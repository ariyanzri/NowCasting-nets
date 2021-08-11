import numpy as np
import h5py
from sklearn.model_selection import train_test_split

def train_test(file):
    
    with h5py.File(file, 'r') as h5:
        X_train = h5['X_train'][:,:,256:,:,:]
        X_valid = h5['X_test'][:,:,256:,:,:]
        y_train = h5['y_train'][:,:,256:,:,:]
        y_valid = h5['y_test'][:,:,256:,:,:]

    return X_train, X_valid, y_train, y_valid
    
def train_test_feedback(file):

    with h5py.File(file, 'r') as h5:
        
        X_train = h5['X_train'][:,:,256:,:,:]
        X_valid = h5['X_test'][:,:,256:,:,:]
        y_train = h5['y_train'][:,:,256:,:,:]
        y_valid = h5['y_test'][:,:,256:,:,:]
        z_train = h5['z_train'][:,:,256:,:,:]
        z_valid = h5['z_test'][:,:,256:,:,:]

    return X_train, X_valid, y_train, y_valid, z_train, z_valid

def train_test_residuals(file, dic):
    with h5py.File(file, 'r') as h5:
        data = h5['Precipitation'][:]
    data = np.expand_dims(data, axis=-1)
    print(data.shape)
    indx = np.arange(0, data.shape[0], dic['input_length'] + dic['output_length'])
    X = [data[indx[i]:indx[i+1]-dic['output_length'], :, :, 0] for i in range(len(indx)-1)]
    X =  np.expand_dims(X, axis=-1)
    Y = [data[indx[i]+dic['input_length']:indx[i+1], :, :, 0] - data[indx[i+1]-dic['output_length']-1, :, :, 0]  for i in range(len(indx)-1)]
    Y =  np.expand_dims(Y, axis=-1)
    Y =  np.expand_dims(Y, axis=-1)
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    X_train = X_train[:, :, 144:400, :256, :] 
#         X_train[:, :, :200, 200:400, :], 
#         X_train[:, :, 200:400, :200, :], 
#         X_train[:, :, 200:400, 200:400, :],
#         X_train[:, :, 400:, :200, :], 
#         X_train[:, :, 400:, 200:400, :]
    
    y_train = y_train[:, :, 144:400, :256, :] 
#         y_train[:, :, :200, 200:400, :], 
#         y_train[:, :, 200:400, :200, :], 
#         y_train[:, :, 200:400, 200:400, :],
#         y_train[:, :, 400:, :200, :], 
#         y_train[:, :, 400:, 200:400, :]'
    
    X_valid = X_valid[:, :, 144:400, :256, :] 
#         X_valid[:, :, :200, 200:400, :], 
#         X_valid[:, :, 200:400, :200, :], 
#         X_valid[:, :, 200:400, 200:400, :],
#         X_valid[:, :, 400:, :200, :], 
#         X_valid[:, :, 400:, 200:400, :]
    
    y_valid = y_valid[:, :, 144:400, :256, :] 
#         y_valid[:, :, :200, 200:400, :], 
#         y_valid[:, :, 200:400, :200, :], 
#         y_valid[:, :, 200:400, 200:400, :],
#         y_valid[:, :, 400:, :200, :], 
#         y_valid[:, :, 400:, 200:400, :]
    percentage = np.array([(X_train[i, 0].squeeze()>1).mean()*100 for i in range(X_train.shape[0])])
    percentage_valid = np.array([(X_valid[i, 0].squeeze()>1).mean()*100 for i in range(X_valid.shape[0])])

    print(X_train.shape, y_valid.shape)
    return X_train[percentage>1], X_valid[percentage_valid>1], y_train[percentage>1], y_valid[percentage_valid>1]

def train_test_experiment1(file, output_length):
    with h5py.File(file, 'r') as h5:
        data = h5['Precipitation'][:]
    data = np.expand_dims(data, axis=-1)
    print(data.shape)
    indx = np.arange(0, data.shape[0], 9 + output_length)
    X = [data[indx[i]:indx[i+1]-output_length, :, :, 0] for i in range(len(indx)-1)]
    X =  np.expand_dims(X, axis=-1)
    Y = [data[indx[i]+9:indx[i+1], :, :, 0] - data[indx[i+1]-output_length-1, :, :, 0]  for i in range(len(indx)-1)]
    Y =  np.expand_dims(Y, axis=-1)
    Y =  np.expand_dims(Y, axis=-1)
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    X_train = X_train[:, :, 144:400, :256, :] 
#         X_train[:, :, :200, 200:400, :], 
#         X_train[:, :, 200:400, :200, :], 
#         X_train[:, :, 200:400, 200:400, :],
#         X_train[:, :, 400:, :200, :], 
#         X_train[:, :, 400:, 200:400, :]
    
    y_train = y_train[:, :, 144:400, :256, :] 
#         y_train[:, :, :200, 200:400, :], 
#         y_train[:, :, 200:400, :200, :], 
#         y_train[:, :, 200:400, 200:400, :],
#         y_train[:, :, 400:, :200, :], 
#         y_train[:, :, 400:, 200:400, :]'
    
    X_valid = X_valid[:, :, 144:400, :256, :] 
#         X_valid[:, :, :200, 200:400, :], 
#         X_valid[:, :, 200:400, :200, :], 
#         X_valid[:, :, 200:400, 200:400, :],
#         X_valid[:, :, 400:, :200, :], 
#         X_valid[:, :, 400:, 200:400, :]
    
    y_valid = y_valid[:, :, 144:400, :256, :] 
#         y_valid[:, :, :200, 200:400, :], 
#         y_valid[:, :, 200:400, :200, :], 
#         y_valid[:, :, 200:400, 200:400, :],
#         y_valid[:, :, 400:, :200, :], 
#         y_valid[:, :, 400:, 200:400, :]
    percentage = np.array([(X_train[i, 0].squeeze()>1).mean()*100 for i in range(X_train.shape[0])])
    percentage_valid = np.array([(X_valid[i, 0].squeeze()>1).mean()*100 for i in range(X_valid.shape[0])])

    print(X_train.shape, y_valid.shape)
    return X_train[percentage>1], X_valid[percentage_valid>1],y_train[percentage>1], y_valid[percentage_valid>1]

def train_test_west(file):
    
    with h5py.File(file, 'r') as h5:
        X_train = h5['X_train'][:,:,:256,:,:]
        y_train = h5['y_train'][:,:,:256,:,:]
        X_valid = h5['X_test'][:,:,:256,:,:]
        y_valid = h5['y_test'][:,:,:256,:,:]

    return  X_train, X_valid, y_train, y_valid

def train_test_west_feedback(file):

    with h5py.File(file, 'r') as h5:
        
        X_train = h5['X_train'][:,:,:256,:,:]
        X_valid = h5['X_test'][:,:,:256,:,:]
        y_train = h5['y_train'][:,:,:256,:,:]
        y_valid = h5['y_test'][:,:,:256,:,:]
        z_train = h5['z_train'][:,:,:256,:,:]
        z_valid = h5['z_test'][:,:,:256,:,:]

    return X_train, X_valid, y_train, y_valid, z_train, z_valid