#get_data.py
#234567890123456789012345678901234567890123456789012345678901234567890123456789

# Imports here
import torch
from torchvision import datasets, transforms

# The command line parser for train.py
def get_dataloaders(data_dir):

# Load the data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
# DONE: Define your transforms for the training, validation, and testing sets
    train_transforms = \
        transforms.Compose([transforms.RandomRotation(30),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])])

    valid_transforms = \
        transforms.Compose([transforms.Resize(255),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])])

    test_transforms = \
        transforms.Compose([transforms.Resize(255),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])])

    data_transforms = {'train': train_transforms, 
                       'valid': valid_transforms, 
                       'test': test_transforms}


# DONE: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    image_datasets = {'train': train_data, 
                      'valid': valid_data, 
                      'test': test_data}
    
# DONE: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, 
                                              batch_size=64, 
                                              shuffle=True)

    validloader = torch.utils.data.DataLoader(valid_data, 
                                              batch_size=64, 
                                              shuffle=True)

    testloader = torch.utils.data.DataLoader(test_data, 
                                             batch_size=64)

    dataloaders = {'train': trainloader, 
                   'valid': validloader, 
                   'test': testloader}

    return(image_datasets, dataloaders)