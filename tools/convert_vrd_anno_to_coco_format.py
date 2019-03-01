#!/usr/bin/env python
# coding: utf-8




import json
import numpy as np
from PIL import Image
from tqdm import tqdm



# [ymin, ymax, xmin, xmax] to [x, y, w, h]
def box_transform(box):
    x = box[2]
    y = box[0]
    w = box[3] - box[2] + 1
    h = box[1] - box[0] + 1
    return [x, y, w, h]



def convert_anno(split):

    with open('data/vrd/new_annotations_' + split + '.json', 'r') as f:
        vrd_anns = json.load(f)


    print(len(vrd_anns))

    img_dir = 'data/vrd/' + split + '_images/'
    new_imgs = []
    new_anns = []
    ann_id = 1
    for f, anns in tqdm(vrd_anns.items()):
        im_w, im_h = Image.open(img_dir + f).size
        image_id = int(f.split('.')[0])
        new_imgs.append(dict(file_name=f, height=im_h, width=im_w, id=image_id))
        # used for duplicate checking
        bbox_set = set()
        for ann in anns:
            # "area" in COCO is the area of segmentation mask, while here it's the area of bbox
            # also need to fake a 'iscrowd' which is always 0
            s_box = ann['subject']['bbox']
            bbox = box_transform(s_box)
            if not tuple(bbox) in bbox_set:
                bbox_set.add(tuple(bbox))
                area = bbox[2] * bbox[3]
                cat = ann['subject']['category']
                new_anns.append(dict(area=area, bbox=bbox, category_id=cat, id=ann_id, image_id=image_id, iscrowd=0))
                ann_id += 1

            o_box = ann['object']['bbox']
            bbox = box_transform(o_box)
            if not tuple(bbox) in bbox_set:
                bbox_set.add(tuple(bbox))
                area = bbox[2] * bbox[3]
                cat = ann['object']['category']
                new_anns.append(dict(area=area, bbox=bbox, category_id=cat, id=ann_id, image_id=image_id, iscrowd=0))
                ann_id += 1

    with open('data/vrd/objects.json', 'r') as f:
        vrd_objs = json.load(f)


    new_objs = []
    for i, obj in enumerate(vrd_objs):
        new_objs.append(dict(id=i, name=obj, supercategory=obj))


    new_data = dict(images=new_imgs, annotations=new_anns, categories=new_objs)

    with open('data/vrd/detections_' + split + '.json', 'w') as outfile:
        json.dump(new_data, outfile)



if __name__ == '__main__':
    convert_anno('train')
    convert_anno('val')
    
