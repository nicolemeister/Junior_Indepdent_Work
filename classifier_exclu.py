import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class ResNet50(nn.Module):
    def __init__(self, n_classes=1000, pretrained=True, hidden_size=2048):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.resnet.fc = nn.Linear(hidden_size, n_classes)

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        outputs = self.resnet(x)
        return outputs

class multilabel_classifier():

    def __init__(self, device, dtype, nclasses=171, modelpath=None, hidden_size=2048, learning_rate=0.1, weight_decay=1e-4, classification_type=2, weights=None, c=20):
        self.nclasses = nclasses
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.weights = torch.FloatTensor(list(weights)).to(device=self.device)
        self.c = int(c)
        
        if modelpath == '/n/fs/context-scr/models/COCOStuff/stage1/standard/model_99.pth':
            self.model = ResNet50(n_classes=171, hidden_size=hidden_size, pretrained=True)
            print('loading 171 classes!')
        else:
            self.model = ResNet50(n_classes=nclasses, hidden_size=hidden_size, pretrained=True)
        
        self.model.require_all_grads()
        self.classification_type = classification_type

        # Multi-GPU training
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

        self.epoch = 1
        self.print_freq = 100

        if modelpath != None:
            A = torch.load(modelpath, map_location=device)
            self.model.load_state_dict(A['model'])
            self.epoch = A['epoch']

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    def save_model(self, path):
        torch.save({'model':self.model.state_dict(), 'optim':self.optimizer, 'epoch':self.epoch}, path)

    def train(self, loader):

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.train()

        loss_list = []
        a = self.classification_type # 4
        for i, (images, labels, ids) in enumerate(loader):
            images = images.to(device=self.device, dtype=self.dtype) 
            labels = labels.to(device=self.device, dtype=torch.long)

            self.optimizer.zero_grad()
            outputs = self.forward(images) # 200x80 
            criterion = nn.CrossEntropyLoss(weight = self.weights)
            loss_sum = 0
            for j in range(self.c):
                loss = criterion(outputs[: , j*a:(j*a)+a], labels[:, j]) # criterion(200x4, 200x1)
                loss_sum += loss
                
            loss_sum.backward()
            self.optimizer.step()
            
            loss_list.append(loss_sum.item())
            if self.print_freq and (i%self.print_freq == 0):
                print('Training epoch {} [{}|{}] loss: {}'.format(self.epoch, i+1, len(loader), loss_sum.item()), flush=True)

        self.epoch += 1
        return loss_list

    def test(self, loader):
        """Evaluate the 'standard baseline' model"""

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        loss_list = []
        a = self.classification_type
        with torch.no_grad():
            #labels_list = np.array([], dtype=np.int).reshape(0, int(self.nclasses/self.classification_type))
            labels_list = np.array([], dtype=np.int).reshape(0, int(self.c))
            scores_list = np.array([], dtype=np.float32).reshape(0, self.nclasses)

            for i, (images, labels, ids) in enumerate(loader):
                images = images.to(device=self.device, dtype=self.dtype)
                labels = labels.to(device=self.device, dtype=torch.long)

                # Center crop
                outputs = self.forward(images)

                # Ten crop
                # bs, ncrops, c, h, w = images.size()
                # outputs = self.forward(images.view(-1, c, h, w)) # fuse batch size and ncrops
                # outputs = outputs.view(bs, ncrops, -1).mean(1) # avg over crops
                scores = torch.sigmoid(outputs).squeeze() # do we still want sigmoid? 
            
                criterion = nn.CrossEntropyLoss(weight = self.weights)
                loss_sum = 0
                for j in range(self.c):
                    loss = criterion(outputs[: , j*a:(j*a)+a], labels[:, j]) # criterion(200x4, 200x1)               
                    loss_sum += loss

                loss_list.append(loss_sum.item())

                labels_list = np.concatenate((labels_list, labels.detach().cpu().numpy()), axis=0)
                scores_list = np.concatenate((scores_list, scores.detach().cpu().numpy()), axis=0)

        return labels_list, scores_list, loss_list

    def get_prediction_examples(self, loader, b):
        """Sorts predictions on b into successful and unsuccessful examples"""

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        success = set()
        failures = set()

        with torch.no_grad():
            for i, (images, labels, ids) in enumerate(loader):
                images = images.to(device=self.device, dtype=self.dtype)
                labels = labels.to(device=self.device, dtype=self.dtype)

                outputs = self.forward(images)
                scores = torch.sigmoid(outputs)
                preds = torch.round(scores).bool()
                for p in range(preds.shape[0]):
                    if preds[p,b] == labels[p,b]:
                        success.add(ids[p])
                    else:
                        failures.add(ids[p])
                print('Minibatch {}/{}: {} failures total, {} successes total'.format(i, len(loader), len(success), len(failures)), flush=True)

        return success, failures
        
