import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import os



class RNNModel(nn.Module):
    def __init__(self,nHidden,nOutput,temperature,batch_size):
        super(RNNModel, self).__init__()
        self.hidden_dim = nHidden #Hidden unit output dimension
        self.output_dim = nOutput #Output softmax dimension
        self.input_dim = nOutput
        self.batch_size = batch_size
        #self.hn = (torch.zeros(1,batch_size,nHidden).float(),torch.zeros(1,batch_size,nHidden).float())
        self.hn = torch.zeros(1,batch_size,nHidden).float()

        self.rnn = nn.RNN(self.input_dim, self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim, self.output_dim)

    def reset_state(self,hn):
        '''Resets the state between minibatches'''
        self.hn = hn

    def forward(self, batch):
        batch,self.hn = self.rnn(batch,self.hn)
        self.hiddenact = batch.clone()
        batch = self.fc1(batch)
        #output = nn.Softmax(batch, dim = )
        return batch,self.hn


