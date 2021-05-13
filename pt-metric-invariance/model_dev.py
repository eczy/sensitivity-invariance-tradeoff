import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import numpy as np
import sampler
import attacks

#         loss, pos_mask, neg_mask = modeling.construct_loss(config, model, inputs, adv_inputs, labels, device)


def construct_loss(config, model, inputs, adv_inputs, labels, device, inv_norm=False): 

    """
    
    """
    loss = 0.0
    adv_inputs = adv_inputs.unsqueeze(1) #torch.Size([256, 1, 28, 28])

    nat_embeddings = model.get_embedding(inputs)
    adv_embeddings = None
    pos_mask = np.inf
    neg_mask = np.inf

    # 