import os
from labelID import REALID,FAKEID
dataset_type='test'
sourcedir='./FAD/clean/test'
labelpath='./FAD/clean/test/test_fasthifigan_lfcc.txt'
labelfile=open(labelpath,'w')

unseen_vocoder=['fasthifigan', 'tacohifigan', 'world','replaceOnceAishell1']
seen_vocoder=['gl','lpcnet','mbmelgan','pwg','straight','stylegan','wavenet','hifigan']

for root, dirs, files in os.walk(sourcedir):
    for f in files: 
        # if  ('nieshuai' in root or 'magicconversa' in root):
        #     continue
        if(f.split('.')[-1]=='wav'):
            if 'fake' in root.split('/')[-2] and '_fasthifigan' in f:
                labelfile.write(f+' '+ os.path.join(sourcedir,'noise606_lfcc_hx', f[:-4]+'.npy')+' %d'%FAKEID+'\n')
            elif 'real' in root.split('/')[-2]:
                if  ('nieshuai' in root or 'magicconversa' in root):
                    labelfile.write(f+' '+ os.path.join(sourcedir,'noise606_lfcc_hx', f[:-4]+'.npy')+' %d'%REALID+'\n')

labelfile.close()
print('finish')
