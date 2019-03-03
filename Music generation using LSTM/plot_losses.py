import numpy as np
import matplotlib.pyplot as plt
import sys

#Plot beautifying parameters
plt.rcParams["lines.linewidth"] = 1.25
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.25
plt.rcParams["errorbar.capsize"] = 1.0


training_losses = []
validation_losses = []

with open(sys.argv[1]) as f:
    for line in f:
        if "Training" in line:
            training_losses.append(float(line.split(":")[1][1:]))
        elif "Validation" in line:
            validation_losses.append(float(line.split(":")[1][1:]))


n_epochs = len(training_losses)
training_losses = np.array(training_losses)
validation_losses = np.array(validation_losses)
plt.plot(np.arange(1,n_epochs+1,1),training_losses,"b",label="Training Loss")
plt.plot(np.arange(1,n_epochs+1,1),validation_losses,"r",label = "Validation Loss")
plt.xlabel(r"\bfseries Epoch")
plt.ylabel(r"\bfseries Losses")
plt.legend()
plt.show()
