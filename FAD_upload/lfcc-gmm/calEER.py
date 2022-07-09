from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import pandas as pd
import numpy as np

unseen_vocoder=['fasthifigan', 'tacohifigan', 'world','replaceOnceAishell1']
seen_vocoder=['gl','lpcnet','mbmelgan','pwg','straight','stylegan','wavenet','hifigan']

scorepath='./FAD_clean/scores-test-seen.txt'
labelpath='./FAD/clean/test/test_seen_gmm_label.txt'
df1=pd.read_table(scorepath,sep=' ',header=None)
df2=pd.read_table(labelpath,sep=' ',header=None)

score=list(df1[1])
tmp=list(df2[1])
label=[1 if i =='real' else 0 for i in tmp]


fpr, tpr, thresholds = roc_curve(label, score, pos_label=1)
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, thresholds)(eer)

print("eer:%.6f"% eer)
print( "thresh:", thresh)
