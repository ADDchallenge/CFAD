import os
from labelID import REALID,FAKEID
sourcedir='./FAD/clean/train'
labelpath='./FAD/clean/train/train_lfcc.txt'

labelfile=open(labelpath,'w')

for root, dirs, files in os.walk(sourcedir):
    for f in files: 
        # if not ('fasthifigan' in root or 'tacohifigan' in root or 'world' in root or 'replaceOnceAishell1' in root or 'nieshuai' in root or 'magicconversa' in root):
        #     continue
        if(f.split('.')[-1]=='wav'):
            types=root.split('/')[-1]
            if 'fake' in root.split('/')[-2]:
                labelfile.write(f+' '+ os.path.join(sourcedir,'lfcc', f[:-4]+'.npy')+' %d '%FAKEID+ types+'\n')
            elif 'real' in root.split('/')[-2]:
                labelfile.write(f+' '+ os.path.join(sourcedir,'lfcc', f[:-4]+'.npy')+' %d '%REALID+ types+'\n')

labelfile.close()
print('finish')
