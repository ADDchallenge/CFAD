from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from torch.nn import functional as F

import os
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data

from lcnn import LCNN
import scipy.io as sio
from labelID import REALID,FAKEID,POSID,NEGID
from utils import datatransform
from dataset import NumpyDataset


if __name__ == '__main__':
        #load yaml file & set comet_ml config
        _abspath = os.path.abspath(__file__)
        dir_yaml = os.path.splitext(_abspath)[0] + '.yaml'
        with open(dir_yaml, 'r') as f_yaml:
                parser = yaml.load(f_yaml)
        
	#set save directory
        save_dir = parser['save_dir'] + parser['name'] + '/'
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        if not os.path.exists(save_dir  + 'results/'):
                os.makedirs(save_dir + 'results/')
        if not os.path.exists(save_dir  + 'models/'):
                os.makedirs(save_dir + 'models/')

        #device setting
        
        cuda = torch.cuda.is_available()
        device = torch.device('cuda:%s'%parser['gpu_idx'][0] if cuda else 'cpu')
        torch.cuda.set_device(device)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(parser['gpu_idx'][0])

        #define model
        model = LCNN()

        model.load_state_dict(torch.load(save_dir+'models/best.pt', map_location=device))
        model=model.to(device)
        #get utt

        test_label=parser['testlabelfile']
        test_wavlist, test_labellist= datatransform(test_label)

        evalset = NumpyDataset(parser, test_wavlist, test_labellist, is_eval=True)
        evalset_gen = data.DataLoader(evalset,
                batch_size = parser['batch_size'],
                shuffle = False,
                drop_last = False,
                num_workers = parser['num_workers'])
                
        model.eval()
        with torch.set_grad_enabled(False):
                with tqdm(total = len(evalset_gen), ncols = 70) as pbar:
                        y_score1 = [] # score for each sample
                        y1 = [] # label for each sample
                        for m_batch, m_label in evalset_gen:
                                m_batch = m_batch.to(device=device,dtype=torch.float)
                                y1.extend(list(m_label))
                                logits1, out1 = model(m_batch)
                                probs = F.softmax(logits1, dim=-1)
                                y_score1.extend([probs[i, FAKEID].item() for i in range(probs.size(0))])
                
                                pbar.update(1)
                #calculate EER
                f_res = open(save_dir + 'results/test_result.txt', 'w')
                for _s, _t in zip(y1, y_score1):
                    f_res.write('{score} {target}\n'.format(score=_s,target=_t))
                f_res.close()
                fpr, tpr, thresholds = roc_curve(y1, y_score1, pos_label=POSID)
                eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
                thresh = interp1d(fpr, thresholds)(eer)
                print("eer:", eer, "thresh:", thresh)