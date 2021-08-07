import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import numpy as np
import sampler
import attacks

def construct_only_nll_ml_invariance(config, model, inputs, invar_inputs, labels, device): 
    # just invariance here
    loss = 0.0
    nll_loss = torch.nn.NLLLoss()
    # invar_inputs = invar_inputs.unsqueeze(1) #torch.Size([256, 1, 28, 28])

    nat_embeddings = model.get_embedding(inputs)
    adv_embeddings = model.get_embedding(invar_inputs)
    outputs = model(inputs)

    metric_loss, norms, pos_mask, neg_mask = sampler.online_mine_angular_hard(labels, nat_embeddings, adv_embeddings, margin=0.05, squared=True, device=device)

    #nll
    xe_loss = nll_loss(outputs, labels)

    a_norm, n_norm, p_norm = norms
    all_norms = a_norm + n_norm + p_norm
    
    #construct loss
    loss = xe_loss + (0.5*metric_loss) + (0.001 * torch.mean(all_norms))

    return loss, pos_mask, neg_mask



def construct_nll_ml_sens_invariance(config, model, inputs, invar_inputs, labels, device): 
    # just sensitivity and invariance here
    loss = 0.0
    nll_loss = torch.nn.NLLLoss()
    invar_inputs = invar_inputs.unsqueeze(1) #torch.Size([256, 1, 28, 28])

    nat_embeddings = model.get_embedding(inputs)
    sens_image = attacks.fgsm_attack(model=model, images=inputs, labels=labels, device=device, eps=0.01)
    sens_embeddings = model.get_embedding(sens_image)
    invar_embeddings = model.get_embedding(invar_inputs)
    outputs = model(inputs)

    sens_metric_loss, invar_norms, pos_mask, neg_mask = sampler.online_mine_angular_hard(labels, nat_embeddings, sens_embeddings, margin=0.05, squared=True, device=device)
    invar_metric_loss, sens_norms, pos_mask, neg_mask = sampler.online_mine_angular_hard(labels, nat_embeddings, invar_embeddings, margin=0.05, squared=True, device=device)

    #nll
    xe_loss = nll_loss(outputs, labels)

    #norms

    all_invar_norms = invar_norms[0] + invar_norms[1] + invar_norms[2]
    all_sens_norms = sens_norms[0] + sens_norms[1] + sens_norms[2]
    
    #construct loss
    loss = xe_loss + (0.5*sens_metric_loss) + (0.5*invar_metric_loss) + (0.001 * torch.mean(all_invar_norms)) (0.001 * torch.mean(all_sens_norms))

    return loss, pos_mask, neg_mask


def construct_loss(config, model, inputs, invar_inputs, labels, device): 
    loss = 0.0
    # invar_inputs = invar_inputs.unsqueeze(1) #torch.Size([256, 1, 28, 28])

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
            invar_embeddings = model.get_embedding(invar_inputs)

    if config.getboolean('x_ent', 'use') is True: 
        nll_loss = torch.nn.NLLLoss()

        if config.get('x_ent', 'perturbation') == "sensitivity":
            inputs = sens_image

        elif config.get('x_ent', 'perturbation') == "invariance":
            inputs = invar_inputs

        outputs = model(inputs)
        xe_loss = nll_loss(outputs, labels)
        
        if config.getboolean('x_ent', 'add_norm_reg') is True:
            lambda_xe = config.getfloat('x_ent', 'lambda')
            
            if config.getboolean('x_ent', 'norm_image_adversarial') is True:
                norm_image = invar_inputs
            else: 
                norm_image = inputs

            xe_reg = lambda_xe * torch.norm(norm_image) 
            xe_loss += xe_reg 

        loss += xe_loss


    # first triplet loss
    if config.getboolean('triplet', 'use') is True: 
        margin = config.getfloat('triplet', 'margin') 
        distance = config.get('triplet', 'distance') 

        if config.get('triplet', 'perturbation') == "sensitivity":
            # print("Using triplet sens...")
            adv_embeddings = sens_embeddings
        
        elif config.get('triplet', 'perturbation') == "invariance":
            # print("Using triplet invariance...")
            adv_embeddings = invar_embeddings

        if distance == "hard angular": 
            metric_loss, norms, pos_mask, neg_mask = sampler.online_mine_angular_hard(labels, nat_embeddings, adv_embeddings, margin=margin, device=device, squared=True)
        
        elif distance == "angular": 
            metric_loss, norms, pos_mask, neg_mask = sampler.online_mine_angular(labels, nat_embeddings, adv_embeddings, margin=margin, device=device, squared=True)

        elif distance == "hard": 
            metric_loss, norms, pos_mask, neg_mask = sampler.online_mine_hard(labels, nat_embeddings, adv_embeddings, angular=False, device=device, squared=True)

        elif distance == "all": 
            metric_loss, norms, pos_mask, neg_mask = sampler.online_mine_all(labels, nat_embeddings, adv_embeddings, angular=False, device=device, squared=True)

        else: 
            raise NotImplementedError
        
        if config.getboolean('triplet', 'as_reg') is True:
            lambda_triplet = config.getfloat('triplet', 'lambda')
            metric_loss *= lambda_triplet 

        loss += metric_loss

        if config.getboolean('triplet', 'reg_embeddings') is True: 
            norm = 0.0
            a_norm, n_norm, p_norm = norms
            norm += a_norm
            norm += n_norm
            norm += p_norm
            
            lambda_reg = config.getfloat('triplet', 'reg_embed_lambda')
            loss += lambda_reg * torch.mean(norm)

    # import pdb; pdb.set_trace()
    # second triplet loss
    if config.getboolean('triplet_additional', 'use') is True: 
        margin = config.getfloat('triplet_additional', 'margin') 
        distance = config.get('triplet_additional', 'distance') 

        if config.get('triplet_additional', 'perturbation') == "sensitivity":
            # print("Using triplet sens...")
            adv_embeddings = sens_embeddings
        
        elif config.get('triplet_additional', 'perturbation') == "invariance":
            # print("Using triplet invariance...")
            adv_embeddings = invar_embeddings

        if distance == "hard angular": 
            metric_loss, norms, pos_mask, neg_mask = sampler.online_mine_angular_hard(labels, nat_embeddings, adv_embeddings, margin=margin, device=device, squared=True)
        
        elif distance == "angular": 
            metric_loss, norms, pos_mask, neg_mask = sampler.online_mine_angular(labels, nat_embeddings, adv_embeddings, margin=margin, device=device, squared=True)

        elif distance == "hard": 
            metric_loss, norms, pos_mask, neg_mask = sampler.online_mine_hard(labels, nat_embeddings, adv_embeddings, angular=False, device=device, squared=True)

        elif distance == "all": 
            metric_loss, norms, pos_mask, neg_mask = sampler.online_mine_all(labels, nat_embeddings, adv_embeddings, angular=False, device=device, squared=True)

        else: 
            raise NotImplementedError
        
        if config.getboolean('triplet_additional', 'as_reg') is True:
            lambda_triplet = config.getfloat('triplet', 'lambda')
            metric_loss *= lambda_triplet 

        loss += metric_loss

        if config.getboolean('triplet_additional', 'reg_embeddings') is True: 
            norm = 0.0
            a_norm, n_norm, p_norm = norms

            if config.getboolean('triplet_additional', 'reg_only_adv') is True: #TODO prbly a sexier way to do this
                norm += a_norm
            else: 
                norm += a_norm
                norm += n_norm
                norm += p_norm

            lambda_reg = config.getfloat('triplet_additional', 'reg_embed_lambda')
            loss += lambda_reg * torch.mean(norm)

    # print("********************SANITY CHECK********************")
    # print(locals())
    # import pdb; 
    # pdb.set_trace()
    return loss, pos_mask, neg_mask