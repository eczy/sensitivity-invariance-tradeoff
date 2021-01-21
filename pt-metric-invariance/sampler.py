import numpy as np 
  
import os
import random
from itertools import permutations
import numpy as np
import torch

RANDOM_SEED=199

def offline_batching(x, y, test_frac=0.3, ap_pairs=10, an_pairs=10):
    np.random.seed(RANDOM_SEED)
    data_xy = tuple([x,y])

    train_frac = 1 - test_frac
    triplet_train_pairs = []
    triplet_test_pairs = []

    for data_class in sorted(set(data_xy[1])): 
        # for each class fetch instances of same / diff class
        same_class_idx = np.where((data_xy[1] == data_class))[0]
        diff_class_idx = np.where(data_xy[1] != data_class)[0]

        # create anchor-positive pairs

        #create positive and anchor
        arr_anchor_pos_idxs = np.random.choice(same_class_idx, replace=False, size=(ap_pairs, 2))
        # anchor = zip(tuple(anchor[0, :], anchor[1, :]))

        # create negative
        arr_negative_idxs = np.random.choice(diff_class_idx, replace=False, size=an_pairs)

        # create train and test dataset using cartesian product of arr_anchor_pos_idxs x arr_negative_idxs
        train_size = int(ap_pairs * train_frac)
        test_size = int(ap_pairs * test_frac)

        #train
        for ap_idx in arr_anchor_pos_idxs[:train_size]:
            anchor_idx = ap_idx[0]
            positive_idx = ap_idx[1]
            anchor = data_xy[0][anchor_idx]
            positive = data_xy[0][positive_idx]

            for neg_idx in arr_negative_idxs:
                negative = data_xy[0][neg_idx]
                triplet_train_pairs.append([anchor, positive, negative])               
        #test
        for ap_idx in arr_anchor_pos_idxs[test_size:]:
            anchor_idx = ap_idx[0]
            positive_idx = ap_idx[1]
            anchor = data_xy[0][anchor_idx]
            positive = data_xy[0][positive_idx]

            for neg_idx in arr_negative_idxs:
                negative = data_xy[0][neg_idx]
                triplet_test_pairs.append([anchor, positive, negative]) 
                
    return np.array(triplet_train_pairs), np.array(triplet_test_pairs)
