import numpy as np
import itertools
import json

def generate_exp_UNET(model_name):
    
    bn_values = [True]
    input_length_values = [9]
    output_length_values = [3]
    start_filter_values = [90]
    conv_kernel_size_values = [3]
    deconv_kernel_size_values = [3]    
    epochs_values = [100]
    lr_values = [1e-3]
    patience_values = [100]
    batch_size_values = [4]
    activation_values = ["relu"]
    optimizer_values = ["Adam"]
    loss_values = ["huber"]
    drop_out_rate_values = [0.1]

    experiments = list(itertools.product(bn_values,input_length_values,output_length_values,start_filter_values,conv_kernel_size_values,\
        deconv_kernel_size_values,epochs_values,lr_values,patience_values,batch_size_values,activation_values,optimizer_values,loss_values,drop_out_rate_values))

    final_experiments = {}

    for i,e in enumerate(experiments):
        new_e = list(e)
        
        name = model_name+"_"+str(i)
        final_experiments[name] = {\
            "input_bn":new_e[0],\
            "input_length":new_e[1],\
            "output_length":new_e[2],\
            "start_filter":new_e[3],\
            "conv_kernel_size":new_e[4],\
            "deconv_kernel_size":new_e[5],\
            "epochs":new_e[6],\
            "lr":new_e[7],\
            "patience":new_e[8],\
            "batch_size":new_e[9],\
            "activation":new_e[10],\
            "optimizer":new_e[11],\
            "loss":new_e[12],\
            "dr_rate":new_e[13]
            }

    print(":: Number of total experiments: "+str(len(final_experiments)))

    return final_experiments

def generate_exp_CONVLSTM(model_name):

    input_length_values = [9]
    output_length_values = [3]
    start_filter_values = [48]
    conv_kernel_size_values = [3]
    deconv_kernel_size_values = [3]
    epochs_values = [100]
    lr_values = [1e-3]
    patience_values = [100]
    batch_size_values = [4]
    optimizer_values = ["Adam"]
    loss_values = ["huber"]

    experiments = list(itertools.product(input_length_values,output_length_values,start_filter_values,conv_kernel_size_values,\
        deconv_kernel_size_values,epochs_values,lr_values,batch_size_values,optimizer_values,loss_values,patience_values))

    final_experiments = {}

    for i,e in enumerate(experiments):
        new_e = list(e)
        
        name = model_name+"_"+str(i)
        final_experiments[name] = {\
            "input_length": new_e[0],\
            "output_length": new_e[1],\
            "start_filter":new_e[2],\
            "conv_kernel_size": new_e[3],\
            'deconv_kernel_size':new_e[4],\
            "epochs": new_e[5],\
            "lr": new_e[6],\
            "batch_size": new_e[7],\
            "optimizer": new_e[8],\
            "loss": new_e[9],\
            "patience": new_e[10]\
            }

    print(":: Number of total experiments: "+str(len(final_experiments)))

    return final_experiments

def save_settings(final_experiments,path):

    with open(path,"w") as f:
        json.dump(final_experiments,f)

path = "/xdisk/ericlyons/data/ariyanzarei/IMERGE-Project/settings"

exp_UNET = generate_exp_UNET("UNET")
exp_UNETR = generate_exp_UNET("UNETR")
exp_UNETB = generate_exp_UNET("UNETB")

save_settings(exp_UNET,path+"/experiments_UNET.json")
save_settings(exp_UNETR,path+"/experiments_UNETR.json")
save_settings(exp_UNETB,path+"/experiments_UNETB.json")

exp_CONVLSTM = generate_exp_CONVLSTM("CONVLSTM")
exp_CONVLSTMR = generate_exp_CONVLSTM("CONVLSTMR")
exp_CONVLSTMB = generate_exp_CONVLSTM("CONVLSTMB")

save_settings(exp_CONVLSTM,path+"/experiments_CONVLSTM.json")
save_settings(exp_CONVLSTMR,path+"/experiments_CONVLSTMR.json")
save_settings(exp_CONVLSTMB,path+"/experiments_CONVLSTMB.json")