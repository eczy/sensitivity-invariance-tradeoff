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
import attacks
import utils

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
eps = config['epsilon']
patience = config['patience']
device_no = 0
device = torch.device('cuda:{}'.format(input_args.device))

model_dir = os.path.join(root_dir, model_name)
if input_args.reset and os.path.exists(model_dir): 
    shutil.rmtree(model_dir)
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
    running_train_loss = 0.0
    running_test_loss = 0.0

    for idx, data in enumerate(train_loader):
        inputs, labels  = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        nat_embeddings = model.get_embedding(inputs)
        adv_image = attacks.fgsm_attack(model=model, images=inputs, labels=labels, device=device, margin=margin, eps=eps)
        adv_embeddings = model.get_embedding(adv_image)
        outputs = model(inputs)

        metric_loss, pos_mask, neg_mask = sampler.online_mine_angular_hard(labels, nat_embeddings, adv_embeddings, angular=True, margin=margin, squared=True, reg=True,device=device)
        xe_loss = nll_loss(outputs, labels)
        loss = xe_loss + (0.5*metric_loss)
        loss.backward()

        optimizer.step()
        running_train_loss += loss.item()
        if idx%20 == 0:
            print(f"Training @ epoch = {epoch}, {idx}/{len(train_loader)}, loss = {loss:.5f}, pos_mask = {pos_mask}, neg_mask = {neg_mask}", end='\r')
    train_epoch_loss = running_train_loss / len(train_loader)


    # test - but keep the grad bc we still need to do attacks 
    for idx, data in enumerate(test_loader):
        inputs, labels  = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        nat_embeddings = model.get_embedding(inputs)
        adv_image = attacks.fgsm_attack(model=model, images=inputs, labels=labels, device=device, margin=margin, eps=eps)
        adv_embeddings = model.get_embedding(adv_image)
        outputs = model(inputs)

        # import pdb; pdb.set_trace();
        metric_loss, pos_mask, neg_mask = sampler.online_mine_angular_hard(labels, nat_embeddings, adv_embeddings, angular=True, margin=margin, squared=True, reg=True,device=device)
        xe_loss = nll_loss(outputs, labels)
        loss = xe_loss + (0.5*metric_loss)
        
        #adversarial notes
        #1. get the x_p, x_n --> use our sampler --> our embedding net
        #2. get x_p' --> use an attack --> (pgd/fgsm/online invariance)
        # loss = metric_loss

        running_test_loss += loss.item()
    test_epoch_loss = running_test_loss / len(test_loader)

    # save model/ embedding info
    if test_epoch_loss < best_loss:
        torch.save(model, os.path.join(model_dir, 'model'))
        best_loss = test_epoch_loss
        last_improved = epoch

        #TODO pickle the best embeddings
        #TODO add TSNE plotting
        #TODO tensorboard
    else: 
        patience-=1

    # plot
    if idx%100 == 0:
        utils.tsne_plot(model, device, viz_train_loader, viz_test_loader, mdir=model_dir, iter_idx=idx)

    epoch +=1
    print(f"\nPatience= {patience}, train_epoch_loss = {train_epoch_loss}, test_epoch_loss = {test_epoch_loss}")
    print(" "*100)

# final plot
utils.tsne_plot(model, device, viz_train_loader, viz_test_loader, mdir=model_dir, iter_idx=idx)
print('Finished Training!!')