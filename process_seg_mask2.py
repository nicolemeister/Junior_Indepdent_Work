import torch
import json
import numpy as np
import pickle
import os
from collections import defaultdict

from PIL import Image
from pycocotools.coco import COCO

steps = [0]

if 0 in steps:
    print('\nGetting segmentation info for validation set...', flush=True)
    # Path to COCO images
    cocoImgDir = '/n/fs/visualai-scr/Data/Coco/2014data'
    dataType = 'train2014'
    annFile = '{}/annotations/instances_{}.json'.format(cocoImgDir, dataType)
    cocoImgDir = '{}/{}'.format(cocoImgDir, dataType)

    # Initialize COCO api for instance annotations
    dataset_type = 'instances'
    dataset = json.load(open(annFile, 'r'))
    coco = COCO(annFile)

    # Create COCO ids --> names dictionary
    cocoID_to_humanlabels = {}
    for item in dataset['categories']:
        cocoID_to_humanlabels[item['id']] = item['name']

    # Load utility files
    humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/data/COCOStuff/humanlabels_to_onehot.pkl', 'rb'))
    biased_classes_mapped = pickle.load(open('/n/fs/context-scr/data/COCOStuff/biased_classes_mapped.pkl', 'rb'))

    seg_info = defaultdict(dict)
    num_imgs = len(os.listdir(cocoImgDir))
    for i,img_path in enumerate(os.listdir(cocoImgDir)):
        img_path = os.path.join(cocoImgDir, img_path)
        imgId = int(img_path.split('_')[-1][:-4])
        annIds = coco.getAnnIds(imgIds=imgId, iscrowd=None)
        anns = coco.loadAnns(annIds)

        img = Image.open(img_path)
        width, height = img.size 
        mask = torch.zeros(width, height)

        for ann in anns:
            adj_label = humanlabels_to_onehot[cocoID_to_humanlabels[ann['category_id']]]
            # if image contains b class
            if adj_label in biased_classes_mapped.keys() or (adj_label in biased_classes_mapped.values()):
                assert ann['image_id'] == imgId
                seg_info[imgId][adj_label]=coco.annToMask(ann)

        # Print progress
        percent = int(100 * (i / num_imgs))
        complete = ''.join(['=' for j in range(percent // 5)])
        todo = ''.join([' ' for j in range(20 - percent // 5)])
        print('[{}{}] {}%'.format(complete, todo,  percent), end='\r')

    print('[{}] {}%'.format(''.join('=' for j in range(20)), 100), end='\r')
    print()

    with open('seg_train.pkl', 'wb') as handle:
        pickle.dump(seg_info, handle)

    del dataset
    del coco


