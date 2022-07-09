import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset


___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"


def genSpoof_list( dir_meta,is_train=False,is_eval=False):
    
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
             key, label = line.strip().split('\t')
             file_list.append(key)
             if label == 'gl':
                d_meta[key] = 0 
             elif label == 'hifigan':
                d_meta[key] = 1
             elif label == 'lpcnet':
                d_meta[key] = 2
             elif label == 'mbmelgan':
                d_meta[key] = 3
             elif label == 'pwg':
                d_meta[key] = 4
             elif label == 'straight':
                d_meta[key] = 5
             elif label == 'stylegan':
                d_meta[key] = 6
             elif label == 'mbmelgan':
                d_meta[key] = 7
        return d_meta,file_list
    
    elif(is_eval):
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list

    else:
        for line in l_meta:
             key, label = line.strip().split('\t')
             file_list.append(key)
             if label == 'gl':
                d_meta[key] = 0 
             elif label == 'hifigan':
                d_meta[key] = 1
             elif label == 'lpcnet':
                d_meta[key] = 2
             elif label == 'mbmelgan':
                d_meta[key] = 3
             elif label == 'pwg':
                d_meta[key] = 4
             elif label == 'straight':
                d_meta[key] = 5
             elif label == 'stylegan':
                d_meta[key] = 6
             elif label == 'mbmelgan':
                d_meta[key] = 7
        return d_meta,file_list



def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	
			

class Dataset_ASVspoof2019_train(Dataset):
	def __init__(self, list_IDs, labels, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            self.list_IDs = list_IDs
            self.labels = labels
            self.base_dir = base_dir
            

	def __len__(self):
           return len(self.list_IDs)


	def __getitem__(self, index):
            self.cut=64600 # take ~4 sec audio (64600 samples)
            key = self.list_IDs[index]
            X,fs = librosa.load(self.base_dir +key, sr=16000) 
            X_pad= pad(X,self.cut)
            x_inp= Tensor(X_pad)
            y = self.labels
            return x_inp, y

            
            
class Dataset_ASVspoof2021_eval(Dataset):
	def __init__(self, list_IDs, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               '''
               
            self.list_IDs = list_IDs
            self.base_dir = base_dir
            

	def __len__(self):
            return len(self.list_IDs)


	def __getitem__(self, index):
            self.cut=64600 # take ~4 sec audio (64600 samples)
            key = self.list_IDs[index]
            X, fs = librosa.load(self.base_dir+ key, sr=16000)
            X_pad = pad(X,self.cut)
            x_inp = Tensor(X_pad)
            return x_inp,key   
            
            

                
                
                



