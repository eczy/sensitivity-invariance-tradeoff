import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

import sampler
# # LeNet Model definition
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# # FGSM attack code
# def fgsm_attack(image, epsilon, data_grad):
#     # Collect the element-wise sign of the data gradient
#     sign_data_grad = data_grad.sign()
#     # Create the perturbed image by adjusting each pixel of the input image
#     perturbed_image = image + epsilon*sign_data_grad
#     # Adding clipping to maintain [0,1] range
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)
#     # Return the perturbed image
#     return perturbed_image




def fgsm_attack(model, images, labels, device, margin, eps):  
    # images = images.to(device)
    # labels = labels.to(device)  
    
    #set up for gradient
    images.requires_grad = True
    model.zero_grad()
    outputs = model(images)

    # defined by d on ∇xL(F(x), y)
    nll_loss = torch.nn.NLLLoss()
    loss = nll_loss(outputs, labels)
    loss.backward()

    # print("attacking")
    # import pdb; pdb.set_trace();
    
    attack_images = images + eps*images.grad.detach().sign()    # make sure to detach the gradient or else i backprop here
    attack_images = torch.clamp(attack_images, 0, 1)
    
    
    return attack_images    