import torch
import numpy as np
import pickle as pkl
from model_lstm import *
from oneHotEncoding import *

refDict = pkl.load(open("character_dict.pkl","rb"))
inverseDict = {v:k for k,v in refDict.items()}
n_hidden = 100
n_output = len(refDict)
sequence_length = 100
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
model.load_state_dict(torch.load("RNN_model.pt"))
model = model.to(computing_device)
model.eval()

song_sequence = []

primeString = "<start>"
primer = torch.zeros(len(primeString),minibatch_size,n_output)
#Prime with <start>
for i,char in zip(range(len(primeString)),primeString):
    primer[i,0,refDict[char]] = 1
    song_sequence.append(char)
inputs = primer.to(computing_device)
default_hn = (torch.zeros(1,minibatch_size,n_hidden).float().to(computing_device),torch.zeros(1,minibatch_size,n_hidden).float().to(computing_device))
model.reset_state(default_hn)
while len(song_sequence) < 3000:
    output,hidden = model(inputs)
    if output.shape[0] > 1:
        output = torch.Tensor([output[-1].detach().cpu().numpy()])
    output = output.view(output.shape[0] * output.shape[1],-1).detach().cpu()
    input_position = np.array([[torch.argmax(output)]])
    print(input_position.shape)
    inputs, pad_length = oneHotEncoding(input_position,minibatch_size =  minibatch_size,len_refDict = len(refDict))

    inputs = torch.from_numpy(inputs).float().to(computing_device)

    song_sequence.append(inverseDict[input_position[0][0]])


with open("generated_music.txt","w+") as f1:
    f1.write(''.join(song_sequence))







