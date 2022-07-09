import torch
import logging
import numpy as np
from torch.utils.data import Dataset
import random
import scipy.io as sio

logger = logging.getLogger(__name__)

class NumpyDataset(Dataset):
    def __init__(self, params, feat_list, label_list, is_eval=False):
        self.params = params
        self.is_eval = is_eval
        self.max_frame_length = params['max_frame_length'] if 'max_frame_length' in params else 600
        # logger.info('[NumpyFeature-Reader] Load the Numpy Feature in an offline way!')

        self.feat_list = feat_list
        self.label_list = label_list


    def __getitem__(self, index):
        utt_id, feat_path = self.feat_list[index]
        ####load numpy
        if(feat_path.split('.')[-1]=='npy'):
            feature = np.load(feat_path)
        
        # ####load mat
        elif(feat_path.split('.')[-1]=='mat'):
            feature = sio.loadmat(feat_path)
            feature = feature['res']
        else:
            print('!!!!!ERROR!!!!!!')
            
        feature = torch.FloatTensor(feature)
        # print('-----feature----')
        # print(feature.shape)

        if feature.size(0) > self.max_frame_length:
            if not (self.is_eval):
                startp = np.random.randint(feature.size(0)-self.max_frame_length)
            else:
                startp = (feature.size(0)-self.max_frame_length)//2

            feature = feature[startp:startp+self.max_frame_length]
        else:
            mul = int(np.ceil(self.max_frame_length / feature.size(0)))
            feature = feature.repeat(mul,1)[:self.max_frame_length]

        #feature = feature[:min(self.max_frame_length if self.max_frame_length >0 else feature.size(0), feature.size(0))]
        
        feature_length = feature.size(0)


        if self.label_list is not None:
            label = self.label_list[index]

            return feature, label
        else:
            return feature

    def __len__(self):
        return len(self.feat_list)
