import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import os



class LSTMModel(nn.Module):
    def __init__(self,nHidden,nOutput,temperature,batch_size):
        super(LSTMModel, self).__init__()
        self.hidden_dim = nHidden #Hidden unit output dimension
        self.output_dim = nOutput #Output softmax dimension
        self.input_dim = nOutput
        self.batch_size = batch_size
        self.hn = (torch.zeros(1,batch_size,nHidden).float(),torch.zeros(1,batch_size,nHidden).float())

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)
        self.drop = nn.Dropout(p = 0.5)
        self.fc1 = nn.Linear(self.hidden_dim, self.output_dim)

    def reset_state(self,hn):
        '''Resets the state between minibatches'''
        self.hn = hn

    def forward(self, batch):
        batch,self.hn = self.lstm(batch,self.hn)
        #batch = self.drop(batch)
        self.hiddenact = batch.clone()
        batch = self.fc1(batch)
        #output = nn.Softmax(batch, dim = )
        return batch,self.hn


