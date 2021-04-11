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
from adversarial_invariance import adversarial_dataset as ad
from torch.utils.tensorboard import SummaryWriter


import sampler
import backbone
from losses import TripletLoss
import attacks
import utils

def train(model, train_loader, config): 
    running_train_loss = 0.0
    print_every = config.getint('model', 'print_every')


    for idx, data in enumerate(train_loader):
        inputs, labels, adv_inputs = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        adv_inputs = adv_inputs.to(device)
        
        loss, pos_mask, neg_mask = modeling.construct_loss(config, model, inputs, adv_inputs, labels, device)
        loss.backward()

        optimizer.step()
        running_train_loss += loss.item()
        if idx%print_every == 0:
            print(f"Training @ epoch = {epoch}, {idx}/{len(train_loader)}, loss = {loss:.5f}, pos_mask = {pos_mask}, neg_mask = {neg_mask}", end='\r')
            print()
    train_epoch_loss = running_train_loss / len(train_loader)

    return model, train_epoch_loss

def test(model, test_loader, config, arr_test_configs):
    running_test_loss = 0.0

    print_every = config.getint('model', 'print_every')

    test_losses = []

    for test_config, config_name in zip(arr_test_configs, ['original', 'sensitivity', 'invariance']): # go through diff outputs
        print(f"***********{config_name} test set **********")

        for idx, data in enumerate(test_loader):
            inputs, labels, adv_inputs = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            adv_inputs = adv_inputs.to(device)

            loss, pos_mask, neg_mask = modeling.construct_loss(test_config, model, inputs, adv_inputs, labels, device)

            running_test_loss += loss.item()    
            if idx%print_every == 0:
                print(f"Testing @ epoch = {epoch}, {idx}/{len(test_loader)}, loss = {loss:.5f}, pos_mask = {pos_mask}, neg_mask = {neg_mask}", end='\r')  
                print()
        test_epoch_loss = running_test_loss / len(test_loader)

        test_losses.append(test_epoch_loss)

        
    return model, test_losses


if __name__ == "__main__": 

    #1. parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="config for model", default="./config_train_mnist_sens.ini")
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
    model_dir = os.path.join(root_dir, 'model_runs', start_time_dir)
    if input_args.reset and os.path.exists(model_dir): 
        shutil.rmtree(model_dir)
    if not os.path.exists(model_dir): 
        os.mkdir(model_dir)

    shutil.copyfile(input_args.config, os.path.join(model_dir, 'config.ini'))

    
    # for test set
    normal_config = configparser.ConfigParser()
    normal_config.read('./config_test_mnist_normal.ini')

    sensitivity_config = configparser.ConfigParser()
    sensitivity_config.read('./config_test_mnist_sensitivity.ini')

    invariance_config = configparser.ConfigParser()
    invariance_config.read('./config_test_mnist_invariance.ini')
    arr_test_configs = [normal_config, sensitivity_config, invariance_config]

    #3. prepare the dataset
    mean, std = 0.1307, 0.3081
    # train_dataset = MNIST(root='../data/MNIST',
    #                     train=True, 
    #                     download=True,
    #                     transform=transforms.Compose([
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((mean,), (std,))]))
    # test_dataset = MNIST(root='../data/MNIST', 
    #                     train=False, 
    #                     download=True,
    #                     transform=transforms.Compose([
    #                             transforms.ToTensor(),
    #                             transforms.Normalize((mean,), (std,))]))
    n_classes = 10

    #4.sampler 
    # online
    
    invar_root = '/data/evan/mnist/mnist'
    normal_root = '../data/MNIST'
    train_dataset = ad.AdversarialMNIST(root=normal_root,
                                        adv_root=invar_root,
                                        train=True, 
                                        download=True,
                                        transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((mean,), (std,))]))

    test_dataset = ad.AdversarialMNIST(root=normal_root,
                                        adv_root=invar_root,
                                        train=False, 
                                        download=True,
                                        transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((mean,), (std,))]))
    n_classes = 10

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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
    writer = SummaryWriter(f'runs/{start_time_dir}')

    last_epoch_improved = 0
    epoch = 0
    best_loss = np.inf
    while epoch - last_epoch_improved < patience:
        
        model, train_epoch_loss = train(model, train_loader, config)
        model, arr_test_losses = test(model, test_loader, config, arr_test_configs)

        orig_loss = arr_test_losses[0]
        sensitivity_loss = arr_test_losses[1]
        invariance_loss = arr_test_losses[2]

        # save model/ embedding info
        if invariance_loss < best_loss:
            torch.save(model, os.path.join(model_dir, 'model'))
            best_loss = invariance_loss
            last_epoch_improved = epoch
        else:
            patience-=1

        epoch +=1
        epoch_time = (time.time() - start_time) / 60
        writer.add_scalar('loss_overall/train epoch loss', train_epoch_loss, epoch)  
        writer.add_scalars(f'loss_overall', {
            'test orig epoch loss': orig_loss,
            'test sensitivity epoch loss': sensitivity_loss,
            'test invariance epoch  loss':invariance_loss}, epoch)   

        writer.add_scalar('loss_test/orig epoch loss', orig_loss, epoch)     
        writer.add_scalar('loss_test/sensitivity epoch loss', sensitivity_loss, epoch)     
        writer.add_scalar('loss_test/invariance epoch  loss', invariance_loss, epoch)             

        print(f"\nPatience= {patience}, Time={epoch_time:.5f}, train_epoch_loss = {train_epoch_loss}, test_epoch_loss = {invariance_loss}")
        print(" "*100)


    utils.tsne_plot(model, device, viz_train_loader, viz_test_loader, mdir=model_dir, iter_idx=100)
    total_time = (time.time() - start_time) / 60
    print(f'Finished Training in: {total_time:.5f}!!')