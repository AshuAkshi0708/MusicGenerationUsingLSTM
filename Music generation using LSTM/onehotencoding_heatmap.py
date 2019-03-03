import numpy as np
import torch

refDict = {}

def integerEncoding(char):
    global refDict
    
    if char not in refDict.keys():
        try:
            refDict[char] = max(refDict.values()) + 1
        except:
            refDict[char] = 0
    #print(refDict)
    return refDict[char]

def oneHotEncoding(minibatch,minibatch_size=1,sequence_size=1,len_refDict = 0):
    if not len_refDict:
        len_refDict = len(refDict)
    encoded = np.zeros((sequence_size,minibatch_size,len_refDict))
    for i in range(len(minibatch)):
        for j in range(len(minibatch[i])):
            if minibatch[i][j] >= 0:
                encoded[j,i,minibatch[i][j]] = 1

    pad_length = int(minibatch_size * sequence_size - np.sum(encoded))
    return (encoded,pad_length)

def getMinibatches(inputFile, batch_size, sequence_length):
    character_indices = []
    trainFile =  open("../pa4Data/pa4Data/train.txt","r")
    while True:
        c1 = trainFile.read(1)
        if not c1:
            break
        integerEncoding(c1)
    while True:
        c = inputFile.read(1)
        if not c:
            break
        character_indices.append(integerEncoding(c))
    character_indices = np.array(character_indices)
    #print(character_indices[:200])
    #Create 100 character sequences
    chunks = []
    targetchunks = []
    for i in range(len(character_indices)//sequence_length):
        chunks.append(character_indices[i*sequence_length:(i+1)*sequence_length])
        targetchunks.append(character_indices[(i)*sequence_length+1:(i+1)*sequence_length+1])
    i = len(character_indices)//sequence_length - 1
    chunks.append(character_indices[(i+1)*sequence_length:])
    targetchunks.append(character_indices[(i+1)*sequence_length+1:])
    targetchunks[-1] = np.append(targetchunks[-1],np.array([-99]))

    minibatches = []
    target_minibatches = []

    skip_length = int(np.ceil(len(chunks)/batch_size))
    for i in range((skip_length)):
        minibatches.append([])
        target_minibatches.append([])
    for i in range(len(chunks)):
        minibatches[i%skip_length].append(np.array(chunks[i]))
    for i in range(len(targetchunks)):
        target_minibatches[i%skip_length].append(np.array(targetchunks[i]))




    #for j in range(len(chunks)//batch_size):
     #   minibatches.append(np.array(chunks[j*batch_size:(j+1)*batch_size]))
      #  target_minibatches.append(np.array(targetchunks[j*batch_size:(j+1)*batch_size]))
    #j = len(chunks)//batch_size -1
    #minibatches.append(np.array(chunks[(j+1)*batch_size:]))
    #target_minibatches.append(np.array(targetchunks[(j+1)*batch_size:]))
    minibatches = np.array(minibatches)
    target_minibatches = np.array(target_minibatches)
    return zip(minibatches,target_minibatches)



#Unit test
if __name__ == "__main__":
    f = open("../pa4Data/pa4Data/train.txt","r")
    minibatch_size = 1
    sequence_size = 100
    iterator = getMinibatches(f,minibatch_size,sequence_size)
    loss = torch.nn.CrossEntropyLoss()
    first_minibatch = None
    second_minibatch = None
    for minibatch,target_minibatch in iterator:
        if first_minibatch is None:
            first_minibatch = minibatch.copy()
        elif second_minibatch is None:
            second_minibatch = minibatch.copy()
        minibatch = np.array(minibatch)
        target_minibatch = np.array(target_minibatch)
        if minibatch.size == 0:
            continue
        inputs, padding = oneHotEncoding(minibatch,minibatch_size, sequence_size)
        print(padding)
        inputs = torch.from_numpy(inputs).float().view(minibatch_size * sequence_size,-1)
        try:
            targets = torch.from_numpy(target_minibatch).long().view(target_minibatch.shape[0] * target_minibatch.shape[1])
        except:
            targets = []
            for i in target_minibatch:
                targets.extend(i)
            targets = np.array(targets)
            targets = torch.from_numpy(targets).long()

        try:
            if padding:
                print(loss(inputs[:-padding],targets))
            else:
                print(loss(inputs,targets))
        except:
            if padding:
                print(loss(inputs[:-padding-1],targets[:-1]))
            else:
                print(loss(inputs[:-1],targets[:-1]))


