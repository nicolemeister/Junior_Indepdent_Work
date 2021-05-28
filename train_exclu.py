import pickle, time, argparse, random
from os import path, makedirs
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support

from classifier_exclu import multilabel_classifier
from load_data_exclu import *
from recall import recall3

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--modelpath', type=str, default=None)
    parser.add_argument('--outdir', type=str, default='save')
    parser.add_argument('--nclasses', type=int, default=171)
    parser.add_argument('--labels_train', type=str, default=None)
    parser.add_argument('--labels_val', type=str, default=None)
    parser.add_argument('--nepoch', type=int, default=100)
    parser.add_argument('--train_batchsize', type=int, default=200)
    parser.add_argument('--val_batchsize', type=int, default=170)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--drop', type=int, default=60)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--hs', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--device', default=torch.device('cuda:0'))
    parser.add_argument('--dtype', default=torch.float32)
    parser.add_argument('--classification_type', default=4)
    parser.add_argument('--weighting_type', default=None)
    parser.add_argument('--c', default=20)

    arg = vars(parser.parse_args())
    print('\n', arg, '\n')
    print('\nTraining with {} GPUs'.format(torch.cuda.device_count()))
    
    arg['outdir'] = arg['outdir']+'_' + arg['weighting_type']+'_' + arg['classification_type'] + '_' + str(arg['c'])#+'_skiperson'
    c = int(arg['c'])
    # Set random seed
    random.seed(arg['seed'])
    np.random.seed(arg['seed'])
    torch.manual_seed(arg['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create output directory
    if not path.isdir(arg['outdir']):
        makedirs(arg['outdir'])
    arg['dataset']='COCOStuff'


    # Load utility files
    biased_classes_mapped = pickle.load(open('/n/fs/context-scr/data/{}/biased_classes_mapped.pkl'.format(arg['dataset']), 'rb'))
    if arg['dataset'] == 'COCOStuff':
        unbiased_classes_mapped = pickle.load(open('/n/fs/context-scr/data/{}/unbiased_classes_mapped.pkl'.format(arg['dataset']), 'rb'))
    humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/data/{}/humanlabels_to_onehot.pkl'.format(arg['dataset']), 'rb'))
    onehot_to_humanlabels = dict((y,x) for x,y in humanlabels_to_onehot.items())
    
    # specify how many classes you want (ex: 2 means 2 classes (exclusive, non-exclusive))
    arg['classification_type'] = int(arg['classification_type'])
    # Create data loaders

    # if classification_type > 0 ,create_dataset changes the labels from size 171 --> size 20 * classification_type
    trainset, weights  = create_dataset(arg['dataset'], arg['labels_train'], biased_classes_mapped,
                                        B=arg['train_batchsize'],  classification_type=int(arg['classification_type']), train=True, weighting_type=arg['weighting_type'])
    valset, _ = create_dataset(arg['dataset'], arg['labels_val'], biased_classes_mapped,
                               B=arg['val_batchsize'], classification_type=int(arg['classification_type']), train=False,  weighting_type=arg['weighting_type'])

    # Initialize classifier
    classifier = multilabel_classifier(arg['device'], arg['dtype'], nclasses=int(c * arg['classification_type']),
                                       modelpath=arg['modelpath'], hidden_size=arg['hs'], learning_rate=arg['lr'], classification_type=arg['classification_type'], weights=weights, c=c)
    classifier.epoch = 1 
    classifier.optimizer = torch.optim.SGD(classifier.model.parameters(), lr=arg['lr'], momentum=arg['momentum'], weight_decay=arg['wd'])

    arg['classification_type'] = int(arg['classification_type'])
    if arg['modelpath']:
        classifier.model.resnet.fc = torch.nn.Linear(arg['hs'], c*arg['classification_type'])
        classifier.nclasses = c * arg['classification_type']

    # Start training
    tb = SummaryWriter(log_dir='{}/runs'.format(arg['outdir']))
    start_time = time.time()

    print('\nStarted training at {}\n'.format(start_time))
    for i in range(1, arg['nepoch']+1):

        # Reduce learning rate from 0.1 to 0.01

        train_loss_list = classifier.train(trainset)

        # Save the model
        if (i + 1) % 1 == 0:
            classifier.save_model('{}/model_{}.pth'.format(arg['outdir'], i))

        # Do inference with the model
        labels_list, scores_list  = classifier.test(valset)
        
        # Record train/val loss
        tb.add_scalar('Loss/Train', np.mean(train_loss_list), i)

        # labels_list is nx20. To feed into average_precision_score, must binarize the labels
        n = labels_list.shape[0]
        labels_list_binarized = np.zeros((n * c, arg['classification_type']))
        for x in range(n):
            labels_list_binarized[x*c:(x*c)+c] = preprocessing.label_binarize(labels_list[x], classes = list(np.arange(arg['classification_type'])))
        
        # scores_list n x ctype*20 
        scores_list = scores_list.reshape((n * c, arg['classification_type']))
        # Calculate and record mAP
        APs = []
        Ps, Rs, F1s = [], [], []
        for k in range(arg['classification_type']):

            APs.append(average_precision_score(labels_list_binarized[:,k], scores_list[:,k]))
            prf = precision_recall_fscore_support(labels_list_binarized[:,k], scores_list[:,k].round())
            Ps.append(prf[0])
            Rs.append(prf[1])
            F1s.append(prf[2])

        for b in range(arg['classification_type']):
            print('{}th class AP: {:.2f} Precision (Neg): {:.2f} Precision (Pos): {:.2f} Recall (Neg): {:.2f} Recall (Pos): {:.2f} F1 Score (Neg): {:.2f} F1 Score (Pos): {:.2f}'.format(i, APs[b], Ps[b][0], Ps[b][1], Rs[b][0], Rs[b][1], F1s[b][0], F1s[b][1]))

        mAP = np.nanmean(APs)
        mP, mR, mF1 = np.nanmean(Ps, axis=0), np.nanmean(Rs, axis=0), np.nanmean(F1s, axis=0)
        print('ALL mAP: {:.2f} Precision (Neg): {:.2f} Precision (Pos): {:.2f} Recall (Neg): {:.2f} Recall (Pos): {:.2f} F1 Score (Neg): {:.2f} F1 Score (Pos): {:.2f}'.format(mAP*100., mP[0], mP[1], mR[0], mR[1], mF1[0], mF1[1]))

        tb.add_scalar('mAP/all', mAP*100, i)
        for c in range(len(mP)):
            tb.add_scalar('mP{}/all'.format(c), mP[c], i)
            tb.add_scalar('mR{}/all'.format(c), mR[c], i)
            tb.add_scalar('mF1{}/all'.format(c), mF1[c], i)



        # Print out information
        print('\nEpoch: {}'.format(i))
        print('Loss: train {:.5f}'.format(np.mean(train_loss_list)))
        print('Val mAP: all {} {:.5f}'.format(arg['classification_type'], mAP*100))
        print('Time passed so far: {:.2f} minutes\n'.format((time.time()-start_time)/60.))

    # Print best model and close tensorboard logger
    tb.close()

if __name__ == "__main__":
    main()
