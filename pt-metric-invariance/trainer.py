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

#1. parse input parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="config for model", default="./config_mnist.json")
parser.add_argument('--device', type=str, help="cuda device no.")
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

#4.sampler -- how to create triplets? --> offline or online? #TODO - start with offline 
X_train, X_test = sampler.offline_batching(train_dataset.data.unsqueeze(1).numpy(), train_dataset.targets.numpy(), ap_pairs=100, an_pairs=100, test_frac=0.2)
train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float()), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test).float()), batch_size=batch_size)

#5. set up model 
embedding_net = backbone.EmbeddingNet()
model = backbone.TripletNet(embedding_net)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
loss_fn = TripletLoss(margin=margin)

#6. train
last_epoch_improved = 0
epoch = 0
best_loss = np.inf

while epoch - last_epoch_improved < patience:
    running_train_loss = 0.0
    running_test_loss = 0.0

    # train
    for idx, (data,) in enumerate(train_loader):
        inputs = data.to(device)
        optimizer.zero_grad()
        anchor, positive, negative = model(x1=inputs[:,0], x2=inputs[:,1], x3=inputs[:,2])
        loss = loss_fn(anchor, positive, negative)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
        if idx%20 == 0:
            print(f"Training @ epoch = {epoch}, {idx}/{len(train_loader)}, loss = {loss:.5f}", end='\r')
    train_epoch_loss = running_train_loss / len(train_loader)
    
    # test
    with torch.no_grad():
        for idx, (data,) in enumerate(test_loader):
            inputs = data.to(device)
            anchor, positive, negative = model(x1=inputs[:,0], x2=inputs[:,1], x3=inputs[:,2])
            loss = loss_fn(anchor, positive, negative)
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

    epoch +=1
    print(f"Patience= {patience}, train_epoch_loss = {train_epoch_loss}, test_epoch_loss = {test_epoch_loss}")
    print(" "*100)
print('Finished Training!!')