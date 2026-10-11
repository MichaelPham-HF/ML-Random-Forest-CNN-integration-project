#use your own images, to train your own cnn with opencv (image recognition in deep learning)
import torch 
import torch.nn as nn #modules and classes to build neural networks
import torch.optim as optim #to call optimizers 
import torchvision #datasets for computer vision tasks
import torchvision.transforms as transforms #transforms for preprocessing images
import matplotlib.pyplot as plt #for plotting
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import cv2 #opencv, gotta install

#make the transformation
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

#load dataset
dataset_path = ""
train_dataset = torchvision.datasets.ImageFolder(root = os.path.join(dataset_path, "train"), transform = transform)
test_dataset = torchvision.datasets.ImageFolder(root = os.path.join(dataset_path, "test"), transform = transform)

#find cnn readt datasets on kaggle

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

#build out a model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1) #input image passed into 32 different filters and extract patterns
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1) #extrac t more complex patterns (object parts, textures)
        #pooling layer
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        #fully connected layers
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, len(train_dataset.classes)) # final classification layer
        self.relu = nn.ReLU()