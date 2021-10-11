#train.py
#234567890123456789012345678901234567890123456789012345678901234567890123456789

# Imports here

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict
import json
import time
from PIL import Image
import numpy as np
import os
from get_input_args import get_input_args_train
from get_data import get_dataloaders

# Retrieve the command line arguments
input_args = get_input_args_train()

# Get image data for training, validation, and testing
image_datasets, dataloaders = get_dataloaders(input_args.data_directory)

# DONE: Build and train your network

# Load a pre-trained network (If you need a starting point, 
# the VGG networks work great and are straightforward to use)
model = getattr(models, input_args.arch)(pretrained=True)
model.arch = input_args.arch

# Freeze our feature parameters
for param in model.parameters():
    param.requires_grad = False

#print(model)

# Add classifier
model.classifier = nn.Sequential(OrderedDict([
                                 ('fc1', nn.Linear(25088, 
                                                   input_args.hidden_units)),
                                 ('relu', nn.ReLU()),
                                 ('drop', nn.Dropout(0.2)),
                                 ('fc2', nn.Linear(input_args.hidden_units,
                                                   102)),
                                 ('output', nn.LogSoftmax(dim=1))
                                ]))
#print(model)


# Train the classifier layers using backpropagation using the pre-trained 
# network to get the features
# Track the loss and accuracy on the validation set to determine 
# the best hyperparameters

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
# learning_rate=0.001 hs proven to be good hyperparameter for this data set

learning_rate = input_args.learning_rate
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

device = torch.device("cuda" if 
                      torch.cuda.is_available() and input_args.gpu == "gpu" 
                      else "cpu")

model.to(device)


epochs = input_args.epochs
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in dataloaders['train']:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders['valid']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += \
                        torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(
              f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss:{running_loss/print_every:.3f}.. "
              f"Validation loss:{valid_loss/len(dataloaders['valid']):.3f}.. "
              f"Validation accuracy:{accuracy/len(dataloaders['valid']):.3f}")
            running_loss = 0
            model.train()
print("Done with training")            

# DONE: Save the checkpoint
checkpoint = {'arch': model.arch,
              'classifier': model.classifier,
              'optimizer': optimizer,
              'optimizer_state': optimizer.state_dict(),
              'criterion': criterion,
              'state_dict': model.state_dict(),
              'class_to_idx': image_datasets['train'].class_to_idx,
              'learning_rate': input_args.learning_rate,
              'hidden_units': input_args.hidden_units,
              'epochs': input_args.epochs
             }

torch.save(checkpoint, input_args.save_dir + '/' + 'checkpoint.pth')