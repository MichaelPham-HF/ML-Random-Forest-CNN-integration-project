import torch 
import torch.nn as nn #modules and classes to build neural networks
import torch.optim as optim #to call optimizers 
import torchvision #datasets for computer vision tasks
import torchvision.transforms as transforms #transforms for preprocessing images
import matplotlib.pyplot as plt #for plotting
from torchvision import models
from torch.utils.data import DataLoader, random_split
import multiprocessing
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#data preprocess
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

#load data
train_set = torchvision.datasets.CIFAR10(root = './data', train = True, transform = transform, download = True)

train_size = int(0.8*len(train_set))
test_size = int(0.2*len(train_set))
train_set, test_set = random_split(train_set, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size = 32, shuffle = True)
test_loader = DataLoader(test_set, batch_size = 32, shuffle = False)

#load model
model = models.resnet18(pretrained = True)
model.fc = nn.Linear(model.fc.in_features, 10) #get last 10 categories
model = model.to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

#train model
def train(model, loader, criterion, optimizer, epochs = 3):
    model.train()
    for epoch in range(epochs):
        total_loss, correct = 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        print(f"Loss: {total_loss/len(loader)}, Accuracy: {correct/len(loader.dataset)}")

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
    print(f"Validation Accuracy: {correct/len(loader.dataset)}")

# def train_with_timeout(model, train_loader, criterion, optimizer, epochs=3, timeout=300): 
#     def run_training():
#         train(model, train_loader, criterion, optimizer, epochs)

#     p = multiprocessing.Process(target=run_training)
#     p.start()

#     p.join(timeout)

#     if p.is_alive():
#         p.terminate()
#         p.join()  
#     else:
#         print("\nTraining completed")

# train_with_timeout(model, train_loader, criterion, optimizer, epochs=3, timeout=30)

train(model, train_loader, criterion, optimizer, epochs = 1)
evaluate(model, test_loader)
