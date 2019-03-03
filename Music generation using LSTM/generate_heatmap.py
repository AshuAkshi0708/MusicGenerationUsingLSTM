import torch
import numpy as np
import pickle as pkl
from model_lstm import *
from onehotencoding_heatmap import *
import matplotlib
import matplotlib.pyplot as plt

refDict = pkl.load(open("character_dict.pkl","rb"))
inverseDict = {v:k for k,v in refDict.items()}
n_hidden = 100
n_output = len(refDict)
sequence_length = 900
minibatch_size = 1
temperature = 0.5

use_cuda = torch.cuda.is_available()
if use_cuda:
    computing_device = torch.device("cuda")
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    print("CUDA NOT supported")


model = LSTMModel(n_hidden, n_output, temperature, minibatch_size)
model.load_state_dict(torch.load("./outputs/hidden_100_lr_1e-3_30epochs/RNN_model.pt"))
model = model.to(computing_device)
model.eval()

song_sequence = []

f = open("generated_music.txt","r")
iterator = getMinibatches(f,minibatch_size,sequence_length)

default_hn = (torch.zeros(1,minibatch_size,n_hidden).float().to(computing_device),torch.zeros(1,minibatch_size,n_hidden).float().to(computing_device))
model.reset_state(default_hn)

act_vals = []


for minibatch_count, (minibatch,target_minibatch) in enumerate(iterator,0):
            minibatch,target_minibatch = np.array(minibatch),np.array(target_minibatch)
            if not minibatch.size:
                continue
            inputs, padding = oneHotEncoding(minibatch,minibatch_size, sequence_length)
            print(inputs.shape)
            inputs = torch.from_numpy(inputs).float().to(computing_device)
            outputs = model.forward(inputs)
            act_val = model.hiddenact.detach().cpu().numpy().transpose()
            print("Each act val shape: ",act_val.shape)
            #Each row represents a neuron for a 900 character sequence
            act_val = act_val.reshape((act_val.shape[0]*act_val.shape[1]), act_val.shape[2])
            #act_val contains the 100 neuron activation for 900 character sequence
            print("Each act val shape after reshaping: ",act_val.shape)
            # Appended list of act_vals of all the minibatches
            act_vals.append(act_val)
            

#act_val = act_vals.detach().cpu().numpy()
#print(np.array(act_vals).shape)
#print(act_val.reshape((act_val.shape[0]*act_val.shape[1]), act_val.shape[2]).shape)
f = open("generated_music.txt","r")
ititerator = getMinibatches(f,minibatch_size,sequence_length)
letter_arrays = []
for minibatch_count, (minibatch,target_minibatch) in enumerate(ititerator,0):
    print(np.array(minibatch[0]).shape)
    if np.array(minibatch[0].shape[0]) == sequence_length:
        print("letters:",np.array(minibatch[0]).shape)
        letter_ind_array = np.array(minibatch[0]).transpose().reshape(30,30)
        letter_arrays.append(letter_ind_array)

#Contains the 900 character sequence for all minibatches
letter_arrays = np.array(letter_arrays)
#print(letter_arrays[0])

for i in range(10):
    plt.figure()
    plt.imshow(act_vals[0][i,:].reshape(30,30),cmap = 'coolwarm')
    plt.colorbar()
    #print(act_vals[0][i,:].reshape(30,30).shape)
    for k in range(len(letter_arrays[0])):
        for l in range(len(letter_arrays[0][k])):
            if inverseDict[letter_arrays[0][k,l]] == '\n':
                text = plt.text(l, k, 'nl',
                   ha="center", va="center", color="k")
            elif inverseDict[letter_arrays[0][k,l]] == ' ':
                text = plt.text(l, k, 'sp',
                   ha="center", va="center", color="k")
            else:
                text = plt.text(l, k, inverseDict[letter_arrays[0][k,l]],
                   ha="center", va="center", color="k")
    plt.savefig('./Heatmaps/heatmap'+str(i+1)+'.png')













