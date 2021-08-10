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
    all_labels = np.array([])
    all_res = np.array([])    
    print_every = config.getint('model', 'print_every')


    for idx, data in enumerate(train_loader):
        inputs, labels, invar_inputs = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        invar_inputs = invar_inputs.to(device)
        
        loss, pos_mask, neg_mask = modeling.construct_loss(config, model, inputs, invar_inputs, labels, device)
        loss.backward()

        optimizer.step()
        running_train_loss += loss.item()
        if idx%print_every == 0:
            print(f"Training @ epoch = {epoch}, {idx}/{len(train_loader)}, loss = {loss:.5f}, pos_mask = {pos_mask}, neg_mask = {neg_mask}", end='\r')
            print()
    train_epoch_loss = running_train_loss / len(train_loader)

    return model, train_epoch_loss

def test(model, test_loader, config):
    arr_test_accs = []

    for config_name in ['original', 'sensitivity', 'invariance']: # go through diff outputs
        print(f"***********{config_name} test set **********")
        all_labels = np.array([])
        all_res = np.array([])

        for __path__, data in enumerate(test_loader):
            inputs, labels, invar_inputs = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            invar_inputs = invar_inputs.to(device)

            if config_name == "original": 
                res = model(inputs)
            
            elif config_name == "sensitivity": 
                eps = config.getfloat('sensitivity', 'epsilon')
                sens_images = attacks.fgsm_attack(model=model, images=inputs, labels=labels, device=device, eps=eps)
                res = model(sens_images)

            elif config_name == "invariance": 
                invar_inputs = invar_inputs.unsqueeze(1)
                res = model(invar_inputs)                
            
            # accuracy 
            # import pdb; pdb.set_trace()
            all_labels = np.append(all_labels, labels.cpu().numpy()) # labels
            probs = torch.exp(res)
            preds = torch.argmax(probs, dim=1)
            all_res = np.append(all_res, preds.cpu().numpy()) # labels
        
        macro_accuracy = round(100 * np.mean(all_res == all_labels), 2)
        print(f"Accuracy: {macro_accuracy}")
        arr_test_accs.append(macro_accuracy)
        
    return model, arr_test_accs


if __name__ == "__main__": 

    #1. parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="config for model", default="./configs-dev/clean_nll_norm.ini")
    parser.add_argument('--device', type=str, help="cuda device no.", default=0)
    parser.add_argument('--reset', '-r', action='store_true', help="whether or not to reset model dir.")
    input_args = parser.parse_args()

    print(f"args: {input_args}")

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
        os.makedirs(model_dir)

    shutil.copyfile(input_args.config, os.path.join(model_dir, 'config.ini'))

    #3. prepare the dataset
    mean, std = 0.1307, 0.3081
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
    n_channels = train_dataset[0][0].size()[0]
    model = backbone.EmbeddingNet(channels_in=n_channels)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    loss_fn = TripletLoss(margin=margin)
    nll_loss = torch.nn.NLLLoss()

    #6. TSNE of data before training
    utils.tsne_plot(model, device, viz_train_loader, viz_test_loader, mdir=model_dir, iter_idx=0)

    #7. train
    writer = SummaryWriter(f'model_runs/{start_time_dir}')

    last_epoch_improved = 0
    epoch = 0
    best_acc = -1 * np.inf
    
    while epoch < n_epochs:
        
        model, train_epoch_loss = train(model, train_loader, config)
        model, arr_test_losses = test(model, test_loader, config)

        orig_acc = arr_test_losses[0]
        sensitivity_acc = arr_test_losses[1]
        invariance_acc = arr_test_losses[2]

        # save model/ embedding info based on invariance accuracy
        if invariance_acc > best_acc:
            torch.save(model, os.path.join(model_dir, 'model'))
            best_acc = invariance_acc
            last_epoch_improved = epoch
        else:
            patience-=1

        epoch +=1
        epoch_time = (time.time() - start_time) / 60
        writer.add_scalar('loss_overall/train epoch loss', train_epoch_loss, epoch)  
        writer.add_scalars(f'loss_overall', {
            'test orig epoch accuracy': orig_acc,
            'test sensitivity epoch accuracy': sensitivity_acc,
            'test invariance epoch accuracy':invariance_acc}, epoch)   

        writer.add_scalar('acc_test/orig epoch loss', orig_acc, epoch)     
        writer.add_scalar('acc_test/sensitivity epoch loss', sensitivity_acc, epoch)     
        writer.add_scalar('acc_test/invariance epoch  loss', invariance_acc, epoch)             

        print(f"\nPatience= {patience}, Time={epoch_time:.5f}, train_epoch_loss = {train_epoch_loss}, test_epoch_acc = {invariance_acc}")
        print(" "*100)


    utils.tsne_plot(model, device, viz_train_loader, viz_test_loader, mdir=model_dir, iter_idx=100)
    total_time = (time.time() - start_time) / 60
    print(f'Finished Training in: {total_time:.5f}!!')
