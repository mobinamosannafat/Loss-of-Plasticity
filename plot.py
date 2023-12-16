'''
RUN GUIDE

1. Structure your directories and plot.py file as follows:
    plot.py
    Relu: destination directory for saving Relu plots
    LeakyRelu: destination directory for saving LeakyRelu plots
    CRelu: destination directory for saving CRelu plots
    ReluData: source directory for Rulu data
    LeakyReluData: source directory for LeakyRulu data
    CReluData: source directory for CRulu data

2. python plot.py

'''


from csuite.environments import dancing_catch
from csuite.environments import catch
from torch import nn
import torch
from collections import deque
import numpy as np
import random
import itertools
import pickle
import matplotlib.pyplot as plt
import tqdm
from sklearn import preprocessing
from sklearn import preprocessing as pre


import warnings

warnings.filterwarnings('ignore')


# Catch Rate
def catch_rate_plot(data, fig_name:str, dest_path:str):
    plt.clf()
    fig =plt.figure(figsize=(8, 6))
    x = np.arange(0, len(data)-1, 1)
    plt.plot(data[1:], linewidth=0.5)
    plt.xlabel('Step', fontsize=20)
    plt.ylabel('Catch Rate',fontsize=20)
    # plt.title('Catch Rate')
    plt.legend()
    plt.savefig(dest_path+fig_name, bbox_inches='tight', dpi=200)
    plt.close()

# Mean Loss
def mean_loss_rate_plot(data, fig_name:str, dest_path:str):
    plt.clf()
    fig =plt.figure(figsize=(8, 6))
    x = np.arange(0, len(data)-1, 1)

    xdata = np.array(data).reshape(-1,1)

    ndata = pre.MinMaxScaler().fit_transform(xdata)
    ndata = ndata.reshape(-1)
    # print(ndata,"\n")

    plt.plot(x,ndata[1:], linewidth=0.5)
    plt.xlabel('Step Sample',fontsize=20)
    plt.ylabel('Mean Loss',fontsize=20)
    # plt.title('Mean Loss')
    plt.legend()
    plt.savefig(dest_path+fig_name, bbox_inches='tight', dpi=200)
    plt.close()


#Sampled loss
def sample_mean_loss_rate_plot(data, fig_name:str, dest_path:str):
    plt.clf()
    fig =plt.figure(figsize=(8, 6))
    x = np.arange(0, len(data)-1, 50)
    # # ndata = [float(i)/0.005 for i in data]
    xdata = np.array(data).reshape(-1,1)

    nndata = xdata[x]
    ndata = pre.MinMaxScaler().fit_transform(nndata)
    ndata = ndata.reshape(-1)
    xax = [i for i in range(0,len(ndata))]
    plt.plot(xax,ndata, linewidth=0.5)
    plt.xlabel('Step Sample',fontsize=20)
    plt.ylabel('Mean Loss',fontsize=20)
    plt.title('Mean Loss')
    plt.legend()
    plt.savefig(dest_path+fig_name, bbox_inches='tight', dpi=200)
    plt.close()


# 10 Sample Gradient
def sample_gradient_plot(data, fig_name:str, dest_path:str):
    x = range(0,10,1)
    norm0 = []
    norm1 = []
    norm2 = []

    nnorm0 = []
    nnorm1 = []
    nnorm2 = []

    for ii,i in enumerate(data):
        if ii % 50 ==0:
            norm0.append(torch.norm(i, p=0).item())
            norm1.append(torch.norm(i, p=1).item())
            norm2.append(torch.norm(i, p=2).item())

    nnorm0 = [float(i)/max(norm0) for i in norm0]
    nnorm1 = [float(i)/max(norm1) for i in norm1]
    nnorm2 = [float(i)/max(norm2) for i in norm2]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(16, 12))

    ax1.plot(x, nnorm0)
    ax1.set_title('Hidden Layer Gradient')
    ax1.set_ylabel('Normalized Norm 0',fontsize=20)
    ax1.legend()
    
    ax2.plot(x, nnorm1)
    ax2.set_ylabel('Normalized Norm 1',fontsize=20)
    ax2.legend()
    
    ax3.plot(x, nnorm2)
    ax3.set_ylabel('Normalized Norm 2',fontsize=20)
    ax3.legend()
    
    plt.xlabel('Step Sample',fontsize=20)
    plt.tight_layout()

    
    plt.savefig(dest_path+fig_name, bbox_inches='tight', dpi=200)
    plt.close()

# Gradient
def gradient_plot(data, fig_name:str, dest_path:str):
    x = range(0,len(data),1)
    norm0 = []
    norm1 = []
    norm2 = []

    nnorm0 = []
    nnorm1 = []
    nnorm2 = []

    for ii,i in enumerate(data):
        # if ii % 50 ==0:
        norm0.append(torch.norm(i, p=0).item())
        norm1.append(torch.norm(i, p=1).item())
        norm2.append(torch.norm(i, p=2).item())

    nnorm0 = [float(i)/max(norm0) for i in norm0]
    nnorm1 = [float(i)/max(norm1) for i in norm1]
    nnorm2 = [float(i)/max(norm2) for i in norm2]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(16, 12))

    ax1.plot(x, nnorm0)
    # ax1.set_title('Hidden Layer Gradient')
    ax1.set_ylabel('Normalized Norm 0',fontsize=20)
    ax1.legend()
    
    ax2.plot(x, nnorm1)
    ax2.set_ylabel('Normalized Norm 1',fontsize=20)
    ax2.legend()
    
    ax3.plot(x, nnorm2)
    ax3.set_ylabel('Normalized Norm 2',fontsize=20)
    ax3.legend()
    
    plt.xlabel('Step Sample',fontsize=20)
    plt.tight_layout()
    plt.xlabel('Step Sample')

    
    plt.savefig(dest_path+fig_name, bbox_inches='tight', dpi=200)
    plt.close()

# Activation
def activation_plot(data, fig_name:str, dest_path:str):
    
    
    x = range(0,len(data)-1,1)
    norm0 = []
    nnorm0 = []
    for ii,i in enumerate(data):
        norm0.append(torch.norm(i, p=0).item())

    nnorm0 = [(float(i)/2050) for i in norm0]

   

    rnnorm0 = [ 1- (float(i)/2050) for i in norm0]

    # print(nnorm0[0],rnnorm0[0])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 12))

    ax1.plot(x, nnorm0[1:])
    # ax1.set_title('Activations')
    ax1.set_ylabel('Normalized Norm 0',fontsize=20)
    # ax1.set_ylim(0,1)
    ax1.legend()
    ax2.plot(x, rnnorm0[1:])
    ax2.set_ylabel('1 - Normalized Norm 1',fontsize=20)
    # ax2.set_ylim(0,1)
    ax2.legend()


    plt.xlabel('Step Sample',fontsize=20)
    # Adjust layout for better spacing
    plt.tight_layout()
    
    plt.savefig(dest_path+fig_name, bbox_inches='tight', dpi=200)
    plt.close()
   
# Weight
def weight_plot(data, fig_name:str, dest_path:str):
    x = range(0,len(data),1)

    norm2 = []
    nnorm2 = []

    for ii,i in enumerate(data):
        norm2.append(torch.norm(i, p=1).item())
    nnorm2 = [float(i)/max(norm2) for i in norm2]


    plt.figure(figsize=(8, 6))
    plt.plot(x, norm2)
    plt.title('Hidden Layer Weights')
    plt.xlabel('Step Sample')
    plt.ylabel('Normalized Norm 2')
    plt.legend()
    
    plt.savefig(dest_path+fig_name, bbox_inches='tight', dpi=200)
    plt.close()
    
# Weight Change
def change_weight_plot(data, fig_name:str, dest_path:str):
    x = range(0,len(data)-1,1)
    dnorm2 = []
    ndnorm2 = []

    for ii, i in enumerate(data):
        try:
            dif = torch.sub(data[ii+1],data[ii])
            dnorm2.append(torch.norm(dif, p=2).item())
        except:
            continue;

    # print(len(dnorm2))
    ndnorm2 = [float(i)/max(dnorm2) for i in dnorm2]


    plt.figure(figsize=(8, 6))
    plt.plot(x, ndnorm2)
    # plt.title('Hidden Layer Weights')
    plt.xlabel('Step Sample',fontsize=20)
    plt.ylabel('Normalized Change Norm 2',fontsize=20)
    plt.legend()
    plt.savefig(dest_path+fig_name, bbox_inches='tight', dpi=200)
    plt.close()

# Main Function
def main ():  

    seeds = [1,2,3,4,5]
    maxstep = 2000000
    swap = 10000
    activations = ['Relu','LeakyRelu','CRelu']
    types = ['hidden_layer_weight','reward','activations','loss','hidden_layer_gradient']
  



    for activation in activations:
        for seed in seeds:
            for data_type in types:

                print(activation,seed,data_type)
                data_path = f"{activation}Data/"
                result_path = f"{activation}/"
                data_name = f"seed{seed}_maxsteps{maxstep}_swap{swap}_{activation}_{data_type}"

                with open(data_path + data_name + ".pkl", 'rb') as f:
                    data = pickle.load(f)

                match data_type:
                    case 'hidden_layer_weight':
                        # weight_plot(data, data_name, result_path )
                        change_weight_plot(data, data_name + '_change',result_path )

                    case 'reward':
                        catch_rate_plot(data, data_name, result_path)

                    case 'activations':
                        activation_plot(data, data_name, result_path )

                    case 'loss':
                        mean_loss_rate_plot(data, data_name, result_path)
                        # sample_mean_loss_rate_plot(data, data_name+'_sample', result_path)


                    case 'hidden_layer_gradient':
                        # sample_gradient_plot(data, data_name + '_sample', result_path)
                        gradient_plot(data, data_name, result_path)

                    case _:
                        print("Invalid data file type.")
                        exit()


if __name__ == "__main__":
    main()                    



