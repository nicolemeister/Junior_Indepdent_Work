import numpy as np
import PIL
import torch
import sys
import os
import cv2
import argparse
import matplotlib.pyplot as plt
import torchvision.transforms as T

from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

from classifier import multilabel_classifier
from load_data import *

from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict
import itertools
from shapely.geometry import box, Polygon
from pycocotools import mask
from skimage import measure


def zero_out_CAM(heatmap, zero_out, binarize=True):
    threshold = np.percentile(heatmap, zero_out)
    # default is binarizing the CAM so that the values below the threshold are 0 and values larger than threshold are 1
    if binarize:
      return heatmap<=threshold

def get_heatmap(CAM_map, img, zero_out_val):
    CAM_map = cv2.resize(CAM_map, (img.shape[0], img.shape[1]))
    CAM_map = CAM_map - np.min(CAM_map)
    CAM_map = CAM_map / np.max(CAM_map)
    CAM_map = 1.0 - CAM_map # make sure colormap is not reversed
    CAM_map = zero_out_CAM(CAM_map, zero_out=zero_out_val)
    return CAM_map


def returnCAM(feature_conv, weight_softmax, class_labels, device):
    bz, nc, h, w = feature_conv.shape # (1, hidden_size, height, width)
    output_cam = torch.Tensor(0, 7, 7).to(device=device)
    for idx in class_labels:
        cam = torch.mm(weight_softmax[idx].unsqueeze(0), feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - cam.min()
        cam_img = cam / cam.max()
        output_cam = torch.cat([output_cam, cam_img.unsqueeze(0)], dim=0)
    return output_cam

def get_cam(img_path, img_labels, b, model, arg, zero_out, biased_classes_mapped, classifier, classifier_features, classifier_softmax_weight):
  normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  transform = T.Compose([
      T.Resize(224),
      T.CenterCrop(224),
      T.ToTensor()
  ])

  
  img_name = img_path.split('/')[-1][:-4]
  #print('img_name: ', img_name)
  original_img = Image.open(img_path).convert('RGB')

  #print('Processing img {}'.format(img_name), flush=True)

  if torch.cuda.device_count() > 0:
      class_labels = img_labels[img_path].type('torch.cuda.ByteTensor')
  else:
      class_labels = img_labels[img_path].type('torch.ByteTensor')
  
  class_labels = torch.flatten(torch.nonzero(class_labels))
  classifier_features.clear()
  img = transform(original_img)
  norm_img = normalize(img)
  norm_img = norm_img.to(device=classifier.device, dtype=classifier.dtype)
  norm_img = norm_img.unsqueeze(0)
  x = classifier.forward(norm_img)
                                                                                                                          
  c = biased_classes_mapped[b]   
  class_labels = torch.tensor([b, c])                                                                                                               
  
  CAMs = returnCAM(classifier_features[0], classifier_softmax_weight, class_labels, arg['device'])
  CAMs = CAMs.detach().cpu().numpy()

  # Save CAM heatmap
  img = np.moveaxis(img.detach().cpu().numpy(), 0, -1)
  class_labels = class_labels.cpu().detach().numpy()
  
  heatmap_b = get_heatmap(CAMs[0], img, zero_out)
  heatmap_c = get_heatmap(CAMs[1], img, zero_out)
  return heatmap_b, heatmap_c


def iou(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 1)   
    mask2_area = np.count_nonzero(mask2 == 1)
    
    intersection = np.count_nonzero( np.logical_and( mask1, mask2) )
    if (mask1_area+mask2_area-intersection)==0:
        return 0
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou

seg_info = pickle.load(open('/n/fs/context-scr/nmeister/SEG_MASK/seg_train.pkl', 'rb'))


# given img id array (either exclu img ids or co-occur img ids)
# return results (avg MSE and avg L2 distance) 
def get_results(img_ids, img_labels, b, arg, biased_classes_mapped, classifier_dict, binarize=False):
  results = defaultdict(int)
  n = len(img_ids)
  print('n: ', n)
  c = biased_classes_mapped[b]
  results['b_s_iou_5_list'] = []
  results['b_s_iou_10_list'] = []
  results['b_w_iou_5_list'] = []
  results['b_w_iou_10_list'] = []
  results['c_s_iou_5_list'] = []
  results['c_s_iou_10_list'] = []
  results['c_w_iou_5_list'] = []
  results['c_w_iou_10_list'] = []
  results['b_size_list'] = []
  results['c_size_list'] = []

  transform = T.Compose([
      T.Resize(224),
      T.CenterCrop(224),
      T.ToTensor()
  ])

  zo_vals = [5, 10] # can try out more zeroing out values

  for img_id in img_ids:
    for zero_out_val in zo_vals:
        #print('image path: ', img_id)
        hmap_b_stnd, hmap_c_stnd = get_cam(img_id, img_labels, b=b, model=arg['modelpath_standard'], arg=arg, zero_out=zero_out_val, biased_classes_mapped=biased_classes_mapped, classifier=classifier_dict['s_classifier'], classifier_features=classifier_dict['s_classifier_features'], classifier_softmax_weight=classifier_dict['s_classifier_softmax_weight'])
        hmap_b_wgtd, hmap_c_wgtd = get_cam(img_id, img_labels, b=b, model=arg['modelpath_weighted'], arg=arg, zero_out=zero_out_val, biased_classes_mapped=biased_classes_mapped, classifier=classifier_dict['w_classifier'], classifier_features=classifier_dict['w_classifier_features'], classifier_softmax_weight=classifier_dict['w_classifier_softmax_weight'])
        num_img_id = int(img_id.split('_')[-1][:-4])

        # get segmentation mask of biased class 
        if b in seg_info[num_img_id].keys():
            b_seg_mask = seg_info[num_img_id][b]
        # if it think there exists a biased class but there is no biased class in the image, skip image
        else: 
            break

        # get segmentation mask of co-occuring class 
        if c in seg_info[num_img_id].keys():
            c_seg_mask = seg_info[num_img_id][c]
        else:
            # create empty binary mask if the co-occur class not in image (ie. exclusive)
            c_seg_mask = np.zeros((224, 224))

        # crop mask to 224x224
        b_seg_mask = transform(Image.fromarray(np.array(b_seg_mask)))
        c_seg_mask = transform(Image.fromarray(np.array(c_seg_mask)))
        b_seg_mask = b_seg_mask > 0
        c_seg_mask = c_seg_mask > 0
        results['b_size_list'].append(b_seg_mask.sum())
        results['c_size_list'].append(c_seg_mask.sum())

        # record IOU of CAM polygon and segmentation mask
        results['b_s_iou_'+str(zero_out_val)+'_list'].append(iou(hmap_b_stnd, b_seg_mask))
        results['b_w_iou_'+str(zero_out_val)+'_list'].append(iou(hmap_b_wgtd, b_seg_mask))
        results['c_s_iou_'+str(zero_out_val)+'_list'].append(iou(hmap_c_stnd, c_seg_mask))
        results['c_w_iou_'+str(zero_out_val)+'_list'].append(iou(hmap_c_wgtd, c_seg_mask))

  return results 

def get_classifier(arg, modelpath):
    classifier_features = []
    def hook_classifier_features(module, input, output):
      classifier_features.append(output)

    classifier = multilabel_classifier(device=arg['device'], dtype=arg['dtype'], modelpath=modelpath)
    classifier.model = classifier.model.to(device=classifier.device, dtype=classifier.dtype)

    classifier.model._modules['resnet'].layer4.register_forward_hook(hook_classifier_features)
    classifier_params = list(classifier.model.parameters())
    classifier_softmax_weight = classifier_params[-2].squeeze(0)

    return classifier, classifier_features, classifier_softmax_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath_standard', type=str, default=None)
    parser.add_argument('--modelpath_weighted', type=str, default=None)
    parser.add_argument('--device', default=torch.device('cuda'))
    parser.add_argument('--dtype', default=torch.float32)
    parser.add_argument('--analysis_type', default='normalized')
    parser.add_argument('--zero_out', default=100) # bottom n% to zero out
    parser.add_argument('--classifier_decision', default='ground_truth')
    parser.add_argument('--binarize', type=bool, default=False)
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--other', type=bool, default=False)
    arg = vars(parser.parse_args())
    print('\n', arg, '\n')

    analysis_type = arg['analysis_type']
    # Get image class labels
    img_labels = pickle.load(open('/n/fs/context-scr/data/COCOStuff/labels_train_20.pkl', 'rb'))

    humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/data/COCOStuff/humanlabels_to_onehot.pkl', 'rb'))
    biased_classes_mapped = pickle.load(open('/n/fs/context-scr/data/COCOStuff/biased_classes_mapped.pkl', 'rb'))
    onehot_to_humanlabels = {v: k for k,v in humanlabels_to_onehot.items()}

    s_classifier, s_classifier_features, s_classifier_softmax_weight = get_classifier(arg,arg['modelpath_standard'])
    w_classifier, w_classifier_features, w_classifier_softmax_weight = get_classifier(arg,arg['modelpath_weighted'])
    print('Got the classifiers..')

    # get a list of co-occur/exclusive images for a specific bc pair to pass into get_results()
    # Create dataloader
    arg['labels'] = '/n/fs/context-scr/data/COCOStuff/labels_train_20.pkl'
    B = 200
    loader = create_dataset('COCOStuff', arg['labels'], biased_classes_mapped, B=B, train=False, splitbiased=False)
    print('Created dataloader...')

    # final results 
    final_c_results=defaultdict(int)
    final_e_results=defaultdict(int)    
    if arg['other']: final_o_results=defaultdict(int)

    classifier_dict = {'s_classifier': s_classifier, 's_classifier_features': s_classifier_features, 's_classifier_softmax_weight': s_classifier_softmax_weight,
                       'w_classifier': w_classifier, 'w_classifier_features': w_classifier_features, 'w_classifier_softmax_weight': w_classifier_softmax_weight}

    
    if arg['classifier_decision'] != 'ground_truth':
        temp_classifier = arg['classifier_decision']
    else:
        temp_classifier = 's_classifier'
    
    labels_list, scores_list, _ = classifier_dict[temp_classifier].test(loader)

    if arg['classifier_decision'] == 'ground_truth':
        scores_list = labels_list
 
    # create ids_exclusive and ids_cooccur which contain img ids for exclusive and co-occur imgs respectively
    for b in biased_classes_mapped.keys():
        c = biased_classes_mapped[b]
        print(onehot_to_humanlabels[b] + ' and ' + onehot_to_humanlabels[c])

        ids_exclusive = []
        ids_cooccur = []
        if arg['other']: ids_other = []
        for i, (images, labels, ids) in enumerate(loader):
            ids = np.array(ids)
            #print('ids.shape: ', ids.shape)
            labels = labels.to(device=arg['device'], dtype=arg['dtype'])
            labels_list = labels.detach().cpu().numpy()
            #print('scores_list.shape: ', scores_list.shape)
            cooccur = (labels_list[:,b]==1) & (labels_list[:,c]==1)
            exclusive = (labels_list[:,b]==1) & (labels_list[:,c]==0)
            #cooccur = (scores_list[i*B:(i+1)*B,b]>0.5) & (scores_list[i*B:(i+1)*B,c]>0.5)
            #exclusive = (scores_list[i*B:(i+1)*B,b]>0.5) & (scores_list[i*B:(i+1)*B,c]<0.5)
            if arg['other']: other = (~exclusive) & (~cooccur)
            #print('exclusive.shape: ', exclusive.shape)
            ids_exclusive.append(ids[exclusive])
            ids_cooccur.append(ids[cooccur])
            if arg['other']: ids_other.append(ids[other])
        
            
        ids_exclusive = list(itertools.chain.from_iterable(ids_exclusive))
        ids_cooccur = list(itertools.chain.from_iterable(ids_cooccur))
        if arg['other']: ids_other = list(itertools.chain.from_iterable(ids_other))

        co_occur_results = get_results(ids_cooccur, img_labels, b, arg, biased_classes_mapped, classifier_dict, binarize=arg['binarize'])
        exclusive_results = get_results(ids_exclusive, img_labels, b, arg, biased_classes_mapped, classifier_dict, binarize=arg['binarize'])
        if arg['other']: other_results = get_results(ids_other, img_labels, b, arg, biased_classes_mapped, classifier_dict, binarize=arg['binarize'])

        #print('co_occur_results: ', co_occur_results)
        #print('exclusive_results: ', exclusive_results)
        final_c_results[b] = co_occur_results
        final_e_results[b] = exclusive_results
        if arg['other']: final_o_results[b] = other_results
    
    # naming convention for output pickles
    if arg['name'] == '':
        name = analysis_type+'_zero_'+str(arg['zero_out'])+'c_decision_'+arg['classifier_decision']+'_binarize_'+str(arg['binarize'])
    else:
        name = arg['name']

    output = open('CAMS/co_occur_results_'+name+'.pkl', 'wb')
    pickle.dump(final_c_results, output)
    output.close()

    output = open('CAMS/exclusive_results_'+name+'.pkl', 'wb')
    pickle.dump(final_e_results, output)
    output.close()
    
    if arg['other']: 
        output = open('CAMS/other_results_'+name+'.pkl', 'wb')
        pickle.dump(final_o_results, output)
        output.close()
    
    return 

if __name__ == "__main__":
    main()
