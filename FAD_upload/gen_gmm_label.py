import os
from labelID import REALID,FAKEID

sourcedir='./FAD/clean/train'
labelpath='./FAD/clean/train/train_gmm_label.txt'


labelfile=open(labelpath,'w')
unseen_vocoder=['fasthifigan', 'tacohifigan', 'world','replaceOnceAishell1']
seen_vocoder=['gl','lpcnet','mbmelgan','pwg','straight','stylegan','wavenet','hifigan']

for root, dirs, files in os.walk(sourcedir):
    for f in files: 
        if ('fasthifigan' in root or 'tacohifigan' in root or 'world' in root or 'replaceOnceAishell1' in root):
            continue
        if( 'nieshuai' in root or 'magicconversa' in root):
            continue
        if(f.split('.')[-1]=='wav'):
            if 'fake' in root.split('/')[-2]:
                labelfile.write(f+' fake '+ os.path.join(root,f)+'\n')
            elif 'real' in root.split('/')[-2]:
                # if ( 'nieshuai' in root or 'magicconversa' in root):
                #     continue
                labelfile.write(f+' real '+ os.path.join(root,f)+'\n')

labelfile.close()
print('finish')
