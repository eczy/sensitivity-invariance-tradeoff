import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class EmbeddingNet(nn.Module):
    def __init__(self, embed_dim=4, num_classes=10):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.embedding = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, embed_dim)
                                )
        
        # TODO is this okay
        # self.step1 = nn.Sequential(nn.Linear(embed_dim, num_classes)), 
        self.fc = nn.Sequential(nn.Linear(embed_dim, num_classes), 
                                nn.LogSoftmax(dim=1)) 


    def get_embedding(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        return self.embedding(output)        

    def forward(self, x):
        output = self.get_embedding(x)

        # import pdb; pdb.set_trace();
        output = self.fc(output)
        return output

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)



# class BasicCNN(nn.Module):
#     def __init__(self, input_shape=(), output_size=4):
#         super(BasicCNN, self).__init__()
#         import pdb;
#         pdb.set_trace()
#         self.cnn1 = nn.Conv2d(input_shape[0], 128, (7,7))
#         self.cnn2 = nn.Conv2d(128, 64, (5,5))
#         self.pooling = nn.MaxPool2d((2,2), (2,2))
#         self.CNN_outshape = self._get_conv_output(input_shape)
#         self.linear = nn.Linear(self.CNN_outshape, output_size)
             
#     def _get_conv_output(self, shape):
#         import pdb; pdb.set_trace() # TODO can i do this without this function
#         bs = 1
#         dummy_x = torch.empty(bs, *shape)
#         x = self._forward_features(dummy_x)
#         CNN_outshape = x.flatten(1).size(1)
#         return CNN_outshape
    
#     def _forward_features(self, x):
#         x = F.tanh(self.cnn1(x))
#         x = self.pooling(x)
#         x = F.tanh(self.cnn2(x))
#         x = self.pooling(x)
#         return x     
        
#     def forward(self, x):
#         x = self._forward_features(x)
#         x = self.linear(x.flatten(1))
#         return x


# class Resnet50(nn.Module): 
#     def __init__(self): 
#         super(Resnet50, self).__init__()
#         self.model = models.resnet50(pretrained=False) #TODO double check if this should be pretrained
#         self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])

#     def forward(self, x):
#         return self.feature_extractor(x)