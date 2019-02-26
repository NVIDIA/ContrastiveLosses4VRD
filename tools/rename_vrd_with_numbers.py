#!/usr/bin/env python
# coding: utf-8

# In[23]:


import json
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import copy
from shutil import copyfile


# take the images from the sg_dataset folder and rename them
# Also converts the gif and png images into jpg

def process_vrd_split(in_split, out_split):
    vrd_dir = 'data/vrd/sg_dataset/sg_' + in_split + '_images/'
    new_dir = 'data/vrd/'+ out_split + '_images/'
    os.mkdir(new_dir)
    
    cnt = 1
    name_map = {}
    for f in tqdm(sorted(os.listdir(vrd_dir))):
    # for f in os.listdir(vrd_dir):
        ext = f.split('.')[1]
        if ext.find('png') >= 0 or ext.find('gif') >= 0:
            img = Image.open(vrd_dir + f).convert('RGB')
        else:        
            copyfile(vrd_dir + f, new_dir + '{:012d}'.format(cnt) + '.jpg')

            
        if ext.find('gif') >= 0:
            img.save(new_dir + '{:012d}'.format(cnt) + '.jpg')
        elif ext.find('png') >= 0:
            img.save(new_dir + '{:012d}'.format(cnt) + '.jpg')
        name_map[f] = cnt
        cnt += 1

    print(len(name_map))


    # store the filename mappings here
    name_map_fname = 'data/vrd/%s_fname_mapping.json' %(out_split)
    with open(name_map_fname, 'w') as f:
        json.dump(name_map, f, sort_keys=True, indent=4)
        f.close()

    # load the original annotations
    with open('data/vrd/annotations_' + in_split + '.json', 'r') as f:
        vrd_anns = json.load(f)
        f.close()
    new_anns = {}
    for k, v in tqdm(vrd_anns.items()):
        # apparently this gif file has been renamed in the original annotations
        if k == '4392556686_44d71ff5a0_o.jpg':
            k = '4392556686_44d71ff5a0_o.gif'
        new_k = '{:012d}'.format(name_map[k]) + '.jpg'
        
        new_anns[new_k] = v


    # create the new annotations 
    with open('data/vrd/new_annotations_' + out_split + '.json', 'w') as outfile:
        json.dump(new_anns, outfile)


if __name__ == '__main__':

    # using the test split as our val. We won't have a true test split for VRD
    process_vrd_split('test', 'val')
    
    process_vrd_split('train', 'train')
