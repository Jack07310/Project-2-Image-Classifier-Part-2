#predict.py
#234567890123456789012345678901234567890123456789012345678901234567890123456789

# Imports here
import json
from get_input_args import get_input_args_predict
import torch
import matplotlib.pyplot as plt
from torchvision import models
from torch import nn
from collections import OrderedDict
from PIL import Image
import numpy as np
import os
from image_utility_functions import process_image, imshow

def predict(image_path, model_path, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # DONE: Implement the code to predict the class from an image file
    
    # Load state from the checkpoint
    state = torch.load(model_path)
    #print(state)
    #print(state['classifier'])

    # Rebild the model
    model = getattr(models, state['arch'])(pretrained=True)
    
    # Freeze our feature parameters
    for param in model.parameters():
        param.requires_grad = False
        
    hidden_units = state['hidden_units']

    # Add classifier
    model.classifier = nn.Sequential(
        OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(0.2)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
    
    model.load_state_dict(state['state_dict'])
    model.classifier = state['classifier']
    model.optimizer = state['optimizer']
    model.optimizer_state = state['optimizer_state']
    model.criterion = state['criterion']
    model.class_to_idx = state['class_to_idx']
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
        
    inputs = process_image(Image.open(image_path))
    inputs = torch.from_numpy(inputs).float()
    #print(inputs.size())
    inputs.unsqueeze_(0)
    #print(inputs.size())

    #print(inputs)
    inputs = inputs.to(device)
    #print(inputs.type)

    labels = os.path.basename(os.path.dirname(image_path))
    #print(labels)

    labels = np.array(labels).astype(np.int)
    #print(labels)

    labels = torch.from_numpy(labels)
    labels.unsqueeze_(0)

    #print(labels)
    labels = labels.to(device)
        
    with torch.no_grad():
        logps = model.forward(inputs)
        #print(labels.size())
        #print(labels)
        batch_loss = model.criterion(logps, labels)
        test_loss = batch_loss.item()
                    
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1, largest=True, sorted=True)
        #print(top_p)
        #print(top_class)
        top_p_np = top_p.detach().cpu().numpy().squeeze()
        #print(top_p_np)
        top_class_indexes = \
            top_class.detach().cpu().numpy().astype(np.int).squeeze().tolist()
        #print(top_class_indexes)
        #print(isinstance(top_class_indexes, int))
        
        # for top_k = 1, top_class_indexes is an integer, not a list
        if topk == 1:
            top_class_indexes = [top_class_indexes]
            
        
        #print(model.class_to_idx)
        top_class_classes = []
        for index in top_class_indexes:
            #print(index)
            key_class = next(key_class for \
                key_class, idx in model.class_to_idx.items() if idx == index)
            #print(key_class)
            #print(cat_to_name[key_class])
            top_class_classes.append(key_class)
        
        #print(top_p_np.astype(float))
        #print(top_class_classes)
        #print([cat_to_name[key_class] for key_class in top_class_classes])
        
        #sort_index = np.argsort(top_p_np[::-1])
        #print(sort_index)
        #print(top_p_np[sort_index])

    return top_p_np.astype(float), top_class_classes


# Retrieve the command line arguments
input_args = get_input_args_predict()

# Load in a mapping from category label to category name
with open(input_args.category_names, 'r') as f:
    cat_to_name = json.load(f)

# DONE: Write a function that loads a checkpoint and rebuilds the model
state = torch.load(input_args.path_to_checkpoint)
#print(state)

# Rebild the model
model = getattr(models, state['arch'])(pretrained=True)

# Freeze our feature parameters
for param in model.parameters():
    param.requires_grad = False

# Add classifier
model.classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(25088, state['hidden_units'])),
                                ('relu', nn.ReLU()),
                                ('drop', nn.Dropout(0.2)),
                                ('fc2', nn.Linear(state['hidden_units'], 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
#print(model)

#Retrive relevant parameters

model.arch = state['arch']
model.classifier = state['classifier']
model.optimizer = state['optimizer']
model.optimizer_state = state['optimizer_state']
model.criterion = state['criterion']
model.load_state_dict(state['state_dict'])
model.class_to_idx = state['class_to_idx']
#model.cat_to_name = state['cat_to_name']
model.learning_rate = state['learning_rate']
model.hidden_units = state['hidden_units']
model.epochs = state['epochs']
#print(model.class_to_idx)



# DONE: Display an image along with the top 5 classes

#image_dir = "1"
#image_path = 'flowers/train/' + image_dir + '/image_06734.jpg'
#image_path = 'flowers/train/' + image_dir + '/image_06735.jpg'
#image_path = 'flowers/train/' + image_dir + '/image_06736.jpg' #misclassified
#image_path = 'flowers/train/' + image_dir + '/image_06737.jpg'
#image_path = 'flowers/train/' + image_dir + '/image_06738.jpg'
#image_path = 'flowers/train/' + image_dir + '/image_06740.jpg'
#image_path = 'flowers/train/' + image_dir + '/image_06741.jpg'
#image_path = 'flowers/train/' + image_dir + '/image_06742.jpg'
#image_path = 'flowers/train/' + image_dir + '/image_06744.jpg'
#image_path = 'flowers/train/' + image_dir + '/image_06745.jpg'
#image_path = 'flowers/train/' + image_dir + '/image_06746.jpg'
#image_path = 'flowers/train/' + image_dir + '/image_06747.jpg'


im = Image.open(input_args.path_to_image)

#for i in range(1,len(cat_to_name)+1):
    #print(i, cat_to_name[str(i)])
    
#print(cat_to_name)
#print(cat_to_name[image_dir])
#print(model.class_to_idx)

image = process_image(im)
#print(image.dtype)

#image_back = imshow(image)
probs, classes = predict(input_args.path_to_image, 
                         input_args.path_to_checkpoint, 
                         input_args.top_k)
#print(probs)
#print(classes)
names = [cat_to_name[key_class] for key_class in classes]

#print(names)
#print(len(names))
#print(probs)

if (len(names) == 1):
    print("the most likely name of this flower is: ", names[0])
    print("with the probability:                    {}".format(probs))
else:
    print("{} top most likely names are: {}".format(input_args.top_k, names))
    print("with the corresponding probabilities: ", probs)
  #            print(f"The most likely name of this flower is: {probs:.3f}")

 #         format() with the probability format ()%")

#fig, (ax1, ax2) = plt.subplots(figsize=(6,7), nrows=2)
#with Image.open(input_args.path_to_image) as im:
#    ax1.imshow(im)
#    ax1.set_title(cat_to_name[image_dir])
#    ax1.axis('off')

#y_pos = np.arange(len(classes))
#ax2.barh(y=y_pos, width=probs)
#ax2.set_yticks(y_pos)
#ax2.set_yticklabels(names)
#ax2.invert_yaxis()