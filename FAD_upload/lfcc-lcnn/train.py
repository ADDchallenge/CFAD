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
from utils import datatransform, datatransform_plus_balance_classes
from dataset import NumpyDataset

if __name__ == '__main__':
    # load yaml file & set comet_ml config
    _abspath = os.path.abspath(__file__)
    dir_yaml = os.path.splitext(_abspath)[0] + '.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser = yaml.load(f_yaml)

    # device setting
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:%s' % parser['gpu_idx'][0] if cuda else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(parser['gpu_idx'][0])

    dev_label=parser['devlabelfile']
    ###!!!!
    dev_wavlist, dev_labellist= datatransform_plus_balance_classes(dev_label, 0)
  

    # define dataset generators
    devset = NumpyDataset(parser, dev_wavlist, dev_labellist, is_eval=True)
    devset_gen = data.DataLoader(devset,
                                 batch_size=parser['batch_size'],
                                 shuffle=False,
                                 drop_last=True,
                                 num_workers=parser['num_workers'])


    # set save directory
    save_dir = parser['save_dir'] + parser['name'] + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir + 'results/'):
        os.makedirs(save_dir + 'results/')
    if not os.path.exists(save_dir + 'models/'):
        os.makedirs(save_dir + 'models/')

    f_params = open(save_dir + 'f_params.txt', 'w')
    for k, v in parser.items():
        print(k, v)
        f_params.write('{}:\t{}\n'.format(k, v))
    f_params.write('LCNN model params\n')


    # define model
    model = LCNN()
    model = model.to(device)

    # set ojbective funtions
    criterion = nn.CrossEntropyLoss()

    # set optimizer
    params = list(model.parameters())
    if parser['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr=parser['lr'],
                                    momentum=parser['opt_mom'],
                                    weight_decay=parser['wd'],
                                    nesterov=bool(parser['nesterov']))

    elif parser['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=parser['lr'],
                                     weight_decay=parser['wd'],
                                     betas=[0.9, 0.98],
                                     eps=1.0e-9,
                                     amsgrad=False)

    ##########################################
    # train/val################################
    ##########################################
    best_eer = 99.
    f_eer = open(save_dir + 'eers.txt', 'a', buffering=1)
    for epoch in tqdm(range(parser['epoch'])):
        f_eer.write('%d ' % epoch)

        # define dataset generators
        x_train, y_train = datatransform_plus_balance_classes(parser['trainlabelfile'], epoch)
        trnset = NumpyDataset(parser, x_train, y_train, is_eval=False)         
        
        trnset_gen = data.DataLoader(trnset,
                                     batch_size=parser['batch_size'],
                                     shuffle=True,
                                     drop_last=True,
                                     num_workers=parser['num_workers'])

        # train phase
        model.train()
        with tqdm(total=len(trnset_gen), ncols=70) as pbar:
            for m_batch, m_label in trnset_gen:
                m_batch, m_label = m_batch.to(device=device,dtype=torch.float), m_label.to(device)

                logits, _ = model(m_batch)
                loss = criterion(logits, m_label)
                optimizer.zero_grad()
                loss.backward()
                model.parameters()
                optimizer.step()

                pbar.set_description('epoch%d:\t loss_ce:%.3f' % (epoch, loss))
                pbar.update(1)
                

        # validation phase
        model.eval()
        with torch.set_grad_enabled(False):
            with tqdm(total=len(devset_gen), ncols=70) as pbar:
                y_score1 = []  # score for each sample
                y1 = []  # label for each sample
                for m_batch, m_label in devset_gen:
                    m_batch= m_batch.to(device=device, dtype=torch.float)
                    y1.extend(list(m_label))
                    logits1, out1 = model(m_batch)
                    probs = F.softmax(logits1, dim=-1)
                    y_score1.extend([probs[i, FAKEID].item() for i in range(probs.size(0))])
                
                    pbar.update(1)

            # calculate EER
            f_res = open(save_dir + 'results/epoch%s.txt' % (epoch), 'w')
            for _s, _t in zip(y1, y_score1):
                f_res.write('{score} {target}\n'.format(score=_s, target=_t))
            f_res.close()
            fpr, tpr, thresholds = roc_curve(y1, y_score1, pos_label=POSID)
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            print(eer)
            
            f_eer.write('%f \n' % eer)


            # record best validation model
            if float(eer) < best_eer:
                print('New best EER: %f' % float(eer))
                best_eer = float(eer)
                dir_best_model_weights = save_dir + 'models/%d-%.6f.h5' % (epoch, eer)
               
                # save best model
                torch.save(model.state_dict(), save_dir + 'models/best.pt')
                print('-----save---')

            if not bool(parser['save_best_only']):
                # save model
                torch.save(model.state_dict(), save_dir + 'models/%d-%.6f.pt' % (epoch, eer))

    f_eer.close()
