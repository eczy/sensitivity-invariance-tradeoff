import torch
import argparse
import json
import os
import shutil
import numpy as np

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

import sampler
import backbone
from losses import TripletLoss

def train(model, train_loader): 
    running_train_loss = 0.0
    running_train_pos_dist = 0.0
    running_train_neg_dist = 0.0    

    for idx, data in enumerate(train_loader):
        # online
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # import pdb; pdb.set_trace()
        outputs = model.embedding_net(inputs)
        loss, pos_dist, neg_dist = sampler.online_mine_angular(labels, outputs, margin=margin, angular=True, squared=True, device=device)
        loss.backward()

        optimizer.step()
        running_train_loss += loss.item()
        running_train_pos_dist += pos_dist.item()
        running_train_neg_dist += neg_dist.item()        
        if idx%20 == 0:
            # offline
            # print(f"Training @ epoch = {epoch}, {idx}/{len(train_loader)}, loss = {loss:.5f}", end='\r')

            #online
            print(f"Training @ epoch = {epoch}, {idx}/{len(train_loader)}, loss = {loss:.5f}, max_pos_dist = {pos_dist:.5f}, max_neg_dist = {neg_dist:.5f}", end='\r')
    train_epoch_loss = running_train_loss / len(train_loader)
    train_pos_dist_epoch = running_train_pos_dist / len(train_loader)
    train_neg_dist_epoch = running_train_neg_dist / len(train_loader)
    return model, train_epoch_loss, train_pos_dist_epoch, train_neg_dist_epoch

def test(model, test_loader):
    running_test_loss = 0.0
    running_test_pos_dist = 0.0
    running_test_neg_dist = 0.0
    
    with torch.no_grad():
        # online
        for idx, data in enumerate(test_loader):
            inputs, labels  = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.embedding_net(inputs)
            loss, pos_dist, neg_dist = sampler.online_mine_angular(labels, outputs, margin=margin, angular=True, squared=True, device=device)
            running_test_loss += loss.item()
            running_test_pos_dist += pos_dist.item()
            running_test_neg_dist += neg_dist.item()             
    test_epoch_loss = running_test_loss / len(test_loader)
    test_pos_dist_epoch = running_test_pos_dist / len(test_loader)
    test_neg_dist_epoch = running_test_neg_dist / len(test_loader) 
    return test_epoch_loss, test_pos_dist_epoch, test_neg_dist_epoch


if __name__ == "__main__": 

    #1. parse input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="config for model", default="./config_mnist.json")
    parser.add_argument('--device', type=str, help="cuda device no.", required=True)
    parser.add_argument('--reset', '-r', action='store_true', help="whether or not to reset model dir.")
    input_args = parser.parse_args()

    #2. setup configs
    with open(input_args.config) as config_file:
        config = json.load(config_file)

    root_dir = config['root_dir']
    model_name = config['model_name']
    lr_rate = config['lr_rate']
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    margin = config['margin']
    patience = config['patience']
    device_no = 0
    device = torch.device('cuda:{}'.format(input_args.device))

    model_dir = os.path.join(root_dir, model_name)
    if input_args.reset and os.path.exists(model_dir): 
        shutil.remove(model_dir)
    if not os.path.exists(model_dir): 
        os.mkdir(model_dir)

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

    # offline
    # X_train, X_test = sampler.offline_batching(train_dataset.data.unsqueeze(1).numpy(), train_dataset.targets.numpy(), ap_pairs=100, an_pairs=100, test_frac=0.2) # X, y --> x_pos, x_anchor, x_neg
    # train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float()), batch_size=batch_size)
    # test_loader = DataLoader(TensorDataset(torch.tensor(X_test).float()), batch_size=batch_size)

    # online
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    #5. set up model 
    embedding_net = backbone.EmbeddingNet()
    model = backbone.TripletNet(embedding_net)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    loss_fn = TripletLoss(margin=margin)
    nll_loss = torch.nn.NLLLoss()


    #7. train
    last_epoch_improved = 0
    epoch = 0
    best_loss = np.inf

    while epoch - last_epoch_improved < patience:
        
        model, train_epoch_loss, train_pos_dist_epoch, train_neg_dist_epoch = train(model, train_loader)
        test_epoch_loss, test_pos_dist_epoch, test_neg_dist_epoch = test(model, test_loader)

        # save model/ embedding info
        if test_epoch_loss < best_loss:
            torch.save(model, os.path.join(model_dir, 'model'))
            best_loss = test_epoch_loss
            last_epoch_improved = epoch
        else:
            patience-=1

        epoch +=1
        print(f"\nPatience= {patience},train_epoch_loss = {train_epoch_loss:.5f}, max_train_pos_dist = {train_pos_dist_epoch:.5f}, max_train_neg_dist = {train_neg_dist_epoch:.5f}\ntest_epoch_loss = {test_epoch_loss:.5f}, max_test_pos_dist = {test_pos_dist_epoch:.5f}, max_test_neg_dist = {test_neg_dist_epoch:.5f}")
        print(" "*100)
    print('Finished Training!!')