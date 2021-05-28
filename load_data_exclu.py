import pickle
import glob
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from skimage import transform

class Dataset(Dataset):
    def __init__(self, img_paths, img_labels, transform=T.ToTensor()):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        ID = self.img_paths[index]
        img = Image.open(ID).convert('RGB')
        X = self.transform(img)
        y = self.img_labels[ID]

        return X, y, ID

def labels_to_class(img_labels, biased_classes_mapped, classification_type=2, weighting_type=None):
    weight = np.zeros(4)
    weight_final = np.zeros(classification_type)
    #biased_classes_mapped = {30: 0} # ski - person
    #biased_classes_mapped = {2: 137}  #car - road
    for i, img_path in enumerate(img_labels):
        labels = np.zeros((len(biased_classes_mapped.keys())))
        for j, b in enumerate(biased_classes_mapped.keys()):
            c = biased_classes_mapped[b]
            # b1c0
            if (img_labels[img_path][b] == 1) and (img_labels[img_path][c] == 0):
                labels[j] = 0
                weight[0]+=1

            # b1c1 
            elif (img_labels[img_path][b] == 1) and (img_labels[img_path][c] == 1):
                labels[j] = 1
                weight[1]+=1

            # b0c0                                                            
            elif (img_labels[img_path][b] == 0) and (img_labels[img_path][c] == 0):
                labels[j] = 2
                weight[2] += 1

            # b0c1
            else:
                labels[j] = 3
                weight[3]+=1
        if classification_type == 2:
            labels[labels>0]=1
            weight_final[0]=weight[0]
            weight_final[1]=weight[1]+weight[2]+weight[3]
        if classification_type == 3:
            for k in np.argwhere(labels==3):
                labels[k] = 2
            weight_final[0]=weight[0]
            weight_final[1]=weight[1]
            weight_final[2] = weight[2]+weight[3]
        if classification_type == 4:
            weight_final=weight
        
        img_labels[img_path] = labels
        

    mapping = {0: 'b1c0', 1: 'b1c1', 2: 'b0c0', 3: 'b0c1'}
    print(weight_final)
    if weighting_type == '1_weight':
        weight_final = weight_final/np.sum(weight_final)
        weight_final = 1-weight_final
    elif weighting_type == 'blog': 
        weight_final = np.sum(weight_final)/(classification_type*weight_final)
    elif weighting_type == 'classbalancing':
        beta=0.999999
        for c in range(classification_type):
            weight_final[c] = (1-beta)/(1-np.power(beta, weight_final[c]))
        sum_weight = np.min(weight_final)
        weight_final /= sum_weight

    print('class distribution: \n')
    for i in range(weight_final.shape[0]):
        print(mapping[i] + ': ' + str(weight_final[i])+'\n')

    return img_labels, weight_final



def create_dataset(dataset, labels_path, biased_classes_mapped, B=100, classification_type=0, train=True, weighting_type=None):

    #img_labels = pickle.load(open(labels_path, 'rb'))
    img_paths = sorted(list(img_labels.keys()))

    # changes the labels from size 171 --> size 20 * classification_type
    if classification_type > 0:
        img_labels, weights = labels_to_class(img_labels, biased_classes_mapped, classification_type=classification_type, weighting_type=weighting_type)

    # Common from here
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        random_resize = True
        if random_resize:
            transform = T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            transform = T.Compose([
                T.Resize(256),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        shuffle = True
    else:
        center_crop = True
        if center_crop:
            transform = T.Compose([
               T.Resize(256),
               T.CenterCrop(224),
               T.ToTensor(),
               normalize
            ])
        else:
            transform = T.Compose([
                T.Resize(256),
                T.TenCrop(224),
                T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops]))
            ])
        shuffle = False

    dset = Dataset(img_paths, img_labels, transform)

    loader = DataLoader(dset, batch_size=B, shuffle=shuffle, num_workers=1)

    return loader, weights

