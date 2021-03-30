import torch
import argparse
import json
import os
import time
import shutil
import configparser
import numpy as np
import modeling

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

import sampler
import backbone
from losses import TripletLoss
import attacks
import utils

def train(model, train_loader, config): 
    running_train_loss = 0.0
    print_every = config.getint('model', 'print_every')

    for idx, data in enumerate(train_loader):
        inputs, labels  = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        loss, pos_mask, neg_mask = modeling.construct_loss(config, model, inputs, labels, device)
        # nat_embeddings = model.get_embedding(inputs)
        # adv_images = attacks.fgsm_attack(model=model, images=inputs, labels=labels, device=device, eps=config.getfloat('sensitivity', 'epsilon'))
        # adv_embeddings = model.get_embedding(adv_images)
        # outputs = model(adv_images)

        # metric_loss, pos_mask, neg_mask = sampler.online_mine_angular_hard(labels, nat_embeddings, adv_embeddings, margin=margin, squared=True, reg=True,device=device)
        # xe_loss = nll_loss(outputs, labels)
        # loss = xe_loss + (0.5*metric_loss)  
        loss.backward()

        optimizer.step()
        running_train_loss += loss.item()
        if idx%print_every == 0:
            print(f"Training @ epoch = {epoch}, {idx}/{len(train_loader)}, loss = {loss:.5f}, pos_mask = {pos_mask}, neg_mask = {neg_mask}", end='\r')
            print()
    train_epoch_loss = running_train_loss / len(train_loader)

    return model, train_epoch_loss

def test(model, test_loader, config):
    running_test_loss = 0.0
    print_every = config.getint('model', 'print_every')

    for idx, data in enumerate(test_loader):
        inputs, labels  = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        loss, pos_mask, neg_mask = modeling.construct_loss(config, model, inputs, labels, device)

        # nat_embeddings = model.get_embedding(inputs)
        # adv_images = attacks.fgsm_attack(model=model, images=inputs, labels=labels, device=device, eps=config.getfloat('sensitivity', 'epsilon'))
        # adv_embeddings = model.get_embedding(adv_images)
        # outputs = model(adv_images)

        # # import pdb; pdb.set_trace();
        # metric_loss, pos_mask, neg_mask = sampler.online_mine_angular_hard(labels, nat_embeddings, adv_embeddings, margin=margin, squared=True, reg=True,device=device)
        # xe_loss = nll_loss(outputs, labels)
        # loss = xe_loss + (0.5*metric_loss)   
        running_test_loss += loss.item()    
        if idx%print_every == 0:
            print(f"Testing @ epoch = {epoch}, {idx}/{len(test_loader)}, loss = {loss:.5f}, pos_mask = {pos_mask}, neg_mask = {neg_mask}", end='\r')  
            print()
    test_epoch_loss = running_test_loss / len(test_loader)
    return model, test_epoch_loss


if __name__ == "__main__": 

    #1. parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="config for model", default="./config_mnist.ini")
    parser.add_argument('--device', type=str, help="cuda device no.", default=0)
    parser.add_argument('--reset', '-r', action='store_true', help="whether or not to reset model dir.")
    input_args = parser.parse_args()

    #2. setup configs
    config = configparser.ConfigParser()
    config.read(input_args.config)

    root_dir = config.get('model', 'root_dir')
    model_name = config.get('model', 'model_name')
    lr_rate = config.getfloat('model', 'lr_rate')
    batch_size = config.getint('model', 'batch_size')
    n_epochs = config.getint('model', 'n_epochs')
    patience = config.getint('model', 'patience')

    margin = config.getfloat('triplet', 'margin')
    device = torch.device('cuda:{}'.format(input_args.device))

    start_time = time.time()
    start_time_dir = f'{model_name}{start_time:0.0f}'
    model_dir = os.path.join(root_dir, start_time_dir)
    if input_args.reset and os.path.exists(model_dir): 
        shutil.rmtree(model_dir)
    if not os.path.exists(model_dir): 
        os.mkdir(model_dir)

    shutil.copyfile(input_args.config, os.path.join(model_dir, 'config.ini'))
    

    #3. prepare the dataset
    mean, std = 0.1307, 0.3081
    train_dataset = MNIST(root='../data/MNIST',
                        train=True, 
                        download=True,
                        transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((mean,), (std,))]))
    test_dataset = MNIST(root='../data/MNIST', 
                        train=False, 
                        download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))]))
    n_classes = 10

    #4.sampler 
    # online
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # data loader for visualization
    viz_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    viz_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #5. set up model 
    model = backbone.EmbeddingNet()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    loss_fn = TripletLoss(margin=margin)
    nll_loss = torch.nn.NLLLoss()

    #6. TSNE of data before training
    utils.tsne_plot(model, device, viz_train_loader, viz_test_loader, mdir=model_dir, iter_idx=0)


    #7. train
    last_epoch_improved = 0
    epoch = 0
    best_loss = np.inf
    while epoch - last_epoch_improved < patience:
        
        model, train_epoch_loss = train(model, train_loader, config)
        model, test_epoch_loss = test(model, test_loader, config)

        # save model/ embedding info
        if test_epoch_loss < best_loss:
            torch.save(model, os.path.join(model_dir, 'model'))
            best_loss = test_epoch_loss
            last_epoch_improved = epoch
        else:
            patience-=1

        epoch +=1
        epoch_time = (time.time() - start_time) / 60
        print(f"\nPatience= {patience}, Time={epoch_time:.5f}, train_epoch_loss = {train_epoch_loss}, test_epoch_loss = {test_epoch_loss}")
        print(" "*100)

    utils.tsne_plot(model, device, viz_train_loader, viz_test_loader, mdir=model_dir, iter_idx=100)
    total_time = (time.time() - start_time) / 60
    print(f'Finished Training in: {total_time:.5f}!!')