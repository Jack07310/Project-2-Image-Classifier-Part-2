#image_utility_functions.py
#234567890123456789012345678901234567890123456789012345678901234567890123456789

# Imports here
#import matplotlib.pyplot as plt
#import torch
#from torch import nn
#import torch.nn.functional as F
from PIL import Image
import numpy as np
#import os

def process_image(image):
    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # DONE: Process a PIL image for use in a PyTorch model
    
    # resize
    #print(image.size)
    width, height = image.size
    #print("width", width)
    #print("height", height)
    if width > height:
        ratio = height / width
        size = 256, ratio * 256
        image.thumbnail(size)
    else:
        ratio = height / width
        size = 256 / ratio, 256
        image.thumbnail(size)
    #print(image.size)

    # crop
    image = image.crop((16, 16, 240, 240))
    #print(image.size)

    
    # convert
    np_image = np.array(image)
    #print(np_image.size)
    #print(np_image.dtype)

    
    # normalize
    #print(np_image[0,:,0])

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = np_image / 255
    
    np_image = (np_image - mean) / std

    # reorded
    #print(np_image.shape)
    np_image = np_image.transpose((2, 0, 1))
    #print(np_image.shape)

    
    return np_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 
    # or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)                      
    ax.set_title(title)
    return ax