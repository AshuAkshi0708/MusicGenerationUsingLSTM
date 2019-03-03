import torch
from model_rnn import *
from oneHotEncoding import *
import pickle

n_epochs = 30
minibatch_size = 1
sequence_length = 100
learning_rate = 0.001
use_cuda = torch.cuda.is_available()
if use_cuda:
    computing_device = torch.device("cuda")
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    print("CUDA NOT supported")

n_hidden = 100
n_output = 93
temperature = 1
n_evalepochs = 1

model = RNNModel(n_hidden,n_output,temperature,minibatch_size)
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate, weight_decay = 1e-2) #TODO - optimizers are defined in the torch.optim package

#default_hn = (torch.zeros(1,minibatch_size,n_hidden).float().to(computing_device),torch.zeros(1,minibatch_size,n_hidden).float().to(computing_device))
default_hn = torch.zeros(1,minibatch_size,n_hidden).float().to(computing_device)
hidden = default_hn
training_losses = []
validation_losses = []
best_val_cost = np.inf
for epoch in range(n_epochs):
    epoch_loss = 0
    model.reset_state(default_hn)
    torch.cuda.empty_cache()
    f = open("../pa4Data/pa4Data/train.txt","r")
    fval = open("../pa4Data/pa4Data/val.txt","r")
    iterator = getMinibatches(f,minibatch_size,sequence_length)
    valiterator = getMinibatches(fval,minibatch_size,sequence_length)
    model.train()
    for minibatch_count,(minibatch,target_minibatch) in enumerate(iterator,0):
        minibatch = np.array(minibatch)
        target_minibatch = np.array(target_minibatch)
        if not minibatch.size:
            continue
        inputs, padding = oneHotEncoding(minibatch,minibatch_size,sequence_length)
        try:
            targets = torch.from_numpy(target_minibatch).long().view(target_minibatch.shape[0] * target_minibatch.shape[1])
        except:
            targets = []
            for i in target_minibatch:
                targets.extend(i)
            targets = np.array(targets)
            targets = torch.from_numpy(targets).long()
        targets = targets.to(computing_device)

        #targets = oneHotEncoding(target_minibatch)
        inputs = torch.from_numpy(inputs).float().to(computing_device)
        #targets = torch.from_numpy(targets).float().to(computing_device)


        # zero accumulated gradients
        optimizer.zero_grad()

        # get the output from the model
        model.reset_state(hidden)
        output,hidden = model.forward(inputs)
        output = output.view(output.shape[0] * output.shape[1],-1)
        #To shove the previous hidden into the next minibatch
        #Without doing backprop across all the previous minibatches
        hidden = hidden.detach()

        #print("Minibatch number",minibatch_count)
        # calculate the loss and perform backprop
        if targets[-1] >= 0:
            if padding:
                loss = criterion(output[:-padding],targets)
            else:
                loss = criterion(output,targets)
        else:
            if padding:
                loss = criterion(output[:-padding-1],targets[:-1])
            else:
                loss = criterion(output[:-1],targets[-1])
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    if epoch % n_evalepochs == 0:
        print("Training Loss per minibatch after epoch ", epoch+1,":", epoch_loss/(minibatch_count + 1))
        training_losses.append(epoch_loss/(minibatch_count+1))
        del targets, minibatch, loss
        torch.cuda.empty_cache()
        val_loss = 0
        model.eval()
        for minibatch_count, (minibatch,target_minibatch) in enumerate(valiterator,0):
            minibatch,target_minibatch = np.array(minibatch),np.array(target_minibatch)
            if not minibatch.size:
                continue
            inputs, padding = oneHotEncoding(minibatch,minibatch_size, sequence_length)
            inputs = torch.from_numpy(inputs).float().to(computing_device)
            try:
                targets = torch.from_numpy(target_minibatch).long().view(target_minibatch.shape[0] * target_minibatch.shape[1])
            except:
                targets = []
                for i in target_minibatch:
                    targets.extend(i)
                targets = np.array(targets)
                targets = torch.from_numpy(targets).long()

            targets = targets.to(computing_device)

            output,hidden = model.forward(inputs)
            output = output.view(output.shape[0] * output.shape[1] , -1)
            hidden =  hidden.detach()
            model.reset_state(hidden)

            if targets[-1] >= 0:
                if padding:
                    loss = criterion(output[:-padding],targets)
                else:
                    loss = criterion(output,targets)
            else:
                if padding:
                    loss = criterion(output[:-padding-1],targets[:-1])
                else:
                    loss = criterion(output[:-1],targets[-1])

            val_loss += loss.item()
        print("Validation Loss after epoch ", epoch+1,":", val_loss/(minibatch_count+1))
        validation_losses.append(val_loss/(minibatch_count+1))
        if validation_losses[-1] < best_val_cost:
            best_val_cost = validation_losses[-1]
            torch.save(model.state_dict(),"Vanilla_model_10_3.pt")

        if epoch == 0:
            pickle_out = open("character_dict.pkl","wb")
            pickle.dump(refDict,pickle_out)
            pickle_out.close()





