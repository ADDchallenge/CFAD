import numpy as np
import torch
from torch.utils import data

def datatransform(labelfile):
    feat_list = []
    label_list = []
    with open(labelfile, 'r', encoding='utf-8') as f:
        for line in f:
            utt_id, feat_path, label = line.strip().split()
            feat_list.append([utt_id,feat_path])
            label_list.append(int(label))

    return feat_list, label_list

def datatransform_plus_balance_classes(labelfile, np_seed):
    '''
    Balance number of sample per class.
    Designed for Binary(two-class) classification.
    '''
    feat_list = []
    label_list = []
    list_0 = []
    list_1 = []
    with open(labelfile, 'r', encoding='utf-8') as f:
        for line in f:
            utt_id, feat_path, label = line.strip().split()
            if(label=='0'):
                list_0.append([utt_id, feat_path, label])
            elif(label=='1'):
                list_1.append([utt_id, feat_path, label])

    lines_small, lines_big = list_0, list_1
    if(len(list_0)>len(list_1)):
        lines_small, lines_big = list_1, list_0

    len_small_lines = len(lines_small)
    np.random.seed(np_seed)
    np.random.shuffle(lines_big)
    new_lines = lines_small + lines_big[:len_small_lines]

    for line in new_lines:
        utt_id, feat_path, label = line
        feat_list.append([utt_id,feat_path])
        label_list.append(int(label))

    return feat_list, label_list
