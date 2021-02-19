import numpy as np 
  
import os
import random
from itertools import permutations
import numpy as np
import torch

"""
This code is a re-implemtnation of KinWaiCheuk/pytorch-triplet-loss

"""

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



#TODO online batching --> cos distance to determine the negative



def _get_triplet_mask(labels, device='cpu'):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: torch.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size()[0]).type(torch.ByteTensor) # make a index in Bool
    indices_not_equal = -(indices_equal-1) # flip booleans
    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = (torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)

    valid_labels = (i_equal_j & ~i_equal_k)
    # Combine the two masks
    
    mask = (distinct_indices.to(device).bool() & valid_labels)

    return mask


def online_mine_all(labels, embeddings, margin, angular, squared=False, device='cpu'):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    if angular: 
        # print("doing angular")
        pairwise_dist = _pairwise_distances_angular(embeddings, squared=squared, device=device)
    else: 
        # Get the pairwise distance matrix
        pairwise_dist = _pairwise_distances(embeddings, squared=squared, device=device)
    # shape (batch_size, batch_size, 1)

    anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels, device=device)
    mask = mask.float()
    
    triplet_loss = mask*triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = torch.max(triplet_loss, torch.Tensor([0.0]).to(device))

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = (triplet_loss > 1e-16).float()
    
    num_positive_triplets = torch.sum(valid_triplets)
    
    num_valid_triplets = torch.sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, torch.mean(anchor_positive_dist), torch.mean(anchor_negative_dist)
    #return triplet_loss, num_positive_triplets, num_valid_triplets

def _get_anchor_positive_triplet_mask(labels, device):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size()[0]).bool().to(device)
    
    indices_not_equal = ~indices_equal # flip booleans

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = (torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1))
    # Combine the two masks
    mask = indices_not_equal & labels_equal

    return mask

def _get_anchor_negative_triplet_mask(labels, device):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = torch.unsqueeze(labels,0) == torch.unsqueeze(labels,1)

    mask = ~labels_equal # invert the boolean tensor

    return mask

def _pairwise_distances(embeddings, squared=False, device='cpu'):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings

    # import pdb; pdb.set_trace();
    embeddings = torch.squeeze(embeddings,1)  # shape=(batch_size, features, 1)
    dot_product = torch.matmul(embeddings, torch.transpose(embeddings,0,1)) # shape=(batch_size, batch_size)
    
    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)
    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = torch.unsqueeze(square_norm, 1) - 2.0 * dot_product + torch.unsqueeze(square_norm, 0)
    
    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.max(distances, torch.Tensor([0.0]).to(device))
    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)
    # return distances matrix (batch X batch)
    return distances


def _pairwise_distances_angular(embeddings, squared=False, device='cpu'):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    embeddings = torch.squeeze(embeddings,1)  # shape=(batch_size, features, 1)
    dot_product = torch.matmul(embeddings, torch.transpose(embeddings,0,1)) # shape=(batch_size, batch_size)
    
    # Compute the pairwise angular distance:
    # cosine dis = dot<a, b> / ||a||^2 * ||b||^2
    # shape (batch_size, batch_size)
    square_element = torch.diag(dot_product)

    a_norm = torch.unsqueeze(square_element, 0) ** 0.5 #(batchsize x 1)
    b_norm = torch.unsqueeze(square_element, 1) ** 0.5
    distances = dot_product / (a_norm * b_norm) # (batchsize x batchsize)

    # import pdb; pdb.set_trace()
    distances = (1 - distances)
    
    return distances    


def online_mine_angular(labels, embeddings, margin, squared, device, reg=True, angular=True): 
    pairwise_dist = _pairwise_distances_angular(embeddings, squared=squared, device=device)

    # get data masks
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels, device)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels, device)

    positive_dist = mask_anchor_positive*pairwise_dist
    negative_dist = mask_anchor_negative*pairwise_dist


    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = torch.max(positive_dist - negative_dist + margin, torch.Tensor([0.0]).to(device))
    
    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss, torch.mean(positive_dist), torch.mean(negative_dist)


def online_mine_hard(labels, embeddings, margin, angular, squared=False, device='cpu'):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """

    if angular: 
        # print("doing angular")
        pairwise_dist = _pairwise_distances_angular(embeddings, squared=squared, device=device)
    else: 
        # Get the pairwise distance matrix
        pairwise_dist = _pairwise_distances(embeddings, squared=squared, device=device)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels, device)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive*pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist = torch.max(anchor_positive_dist, 1, keepdim=True)[0]

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels, device)


    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = torch.max(pairwise_dist, 1, keepdim=True)[0]
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * ~(mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = torch.min(anchor_negative_dist, 1, keepdim=True)[0]

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = torch.max(hardest_positive_dist - hardest_negative_dist + margin, torch.Tensor([0.0]).to(device))
    
    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    # import pdb; pdb.set_trace()

    return triplet_loss, torch.mean(hardest_positive_dist), torch.mean(hardest_negative_dist)
