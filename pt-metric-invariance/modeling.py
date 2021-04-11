import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import numpy as np
import sampler
import attacks


def construct_loss(config, model, inputs, adv_inputs, labels, device): 
    loss = 0.0
    adv_inputs = adv_inputs.unsqueeze(1) #torch.Size([256, 1, 28, 28])

    nat_embeddings = model.get_embedding(inputs)
    adv_embeddings = None
    pos_mask = np.inf
    neg_mask = np.inf

    if config.getboolean('sensitivity', 'use') is True: 
        if config.get('sensitivity', 'mode') == "FGSM":
            eps = config.getfloat('sensitivity', 'epsilon')
            sens_image = attacks.fgsm_attack(model=model, images=inputs, labels=labels, device=device, eps=eps)
            sens_embeddings = model.get_embedding(sens_image)

    if config.getboolean('invariance', 'use') is True: 
        if config.get('invariance', 'mode') == "linf":
            eps = config.getfloat('invariance', 'epsilon')
            # import pdb; pdb.set_trace();
            invar_embeddings = model.get_embedding(adv_inputs)

    if config.getboolean('x_ent', 'use') is True: 
        nll_loss = torch.nn.NLLLoss()

        if config.get('x_ent', 'perturbation') == "sensitivity":
            inputs = sens_image

        elif config.get('x_ent', 'perturbation') == "invariance":
            inputs = adv_inputs

        outputs = model(inputs)
        xe_loss = nll_loss(outputs, labels)
        
        if config.getboolean('x_ent', 'as_reg') is True:
            lambda_xe = config.getfloat('x_ent', 'lambda')
            xe_loss *= lambda_xe 

        loss += xe_loss

    if config.getboolean('triplet', 'use') is True: 
        margin = config.getfloat('triplet', 'margin') 
        reg_embeddings = config.getboolean('triplet', 'reg_embeddings') 
        distance = config.get('triplet', 'distance') 

        if config.get('triplet', 'perturbation') == "sensitivity":
            # print("Using triplet sens...")
            adv_embeddings = sens_embeddings
        
        elif config.get('triplet', 'perturbation') == "invariance":
            # print("Using triplet invariance...")
            adv_embeddings = invar_embeddings

        if distance == "hard angular": 
            metric_loss, pos_mask, neg_mask = sampler.online_mine_angular_hard(labels, nat_embeddings, adv_embeddings, margin=margin, reg=reg_embeddings, device=device, squared=True)
        
        elif distance == "angular": 
            metric_loss, pos_mask, neg_mask = sampler.online_mine_angular(labels, nat_embeddings, adv_embeddings, margin=margin, reg=reg_embeddings, device=device, squared=True)

        elif distance == "hard": 
            metric_loss, pos_mask, neg_mask = sampler.online_mine_hard(labels, nat_embeddings, adv_embeddings, angular=False, device=device, squared=True)

        elif distance == "all": 
            metric_loss, pos_mask, neg_mask = sampler.online_mine_all(labels, nat_embeddings, adv_embeddings, angular=False, device=device, squared=True)

        else: 
            raise NotImplementedError
        
        if config.getboolean('triplet', 'as_reg') is True:
            lambda_triplet = config.getfloat('triplet', 'lambda')
            metric_loss *= lambda_triplet 

        loss += metric_loss

    return loss, pos_mask, neg_mask