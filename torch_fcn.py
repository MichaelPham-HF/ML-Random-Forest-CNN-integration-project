#FCN -  Fully connected neural networks 
import torch 
import torch.nn as nn #modules and classes to build neural networks
import torch.optim as optim #to call optimizers 
import torchvision #datasets for computer vision tasks
import torchvision.transforms as transforms #transforms for preprocessing images
import matplotlib.pyplot as plt #for plotting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #checks if gpu is available, otherwise cpu
print(f"Using: {device}")


#Load, preprocess
transform = transforms.Compose([ #converges images to pytorch tensors, then normalizes pixels to [-1, 1] range
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#MNIST is built-in dataset 28x28 of handwritten digits
train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transform, download = True)
testing_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = transform, download = True)

#create some dataloaders (batch size is 64 for efficient training, shuffle to generalize and improve randomness)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size = 64, shuffle = False)
#testing is not shuffled

#build neural network
class SimpleNN(nn.Module): #fully connected 3 layers
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(28*28, 128) #Input: 784, Output: 128
        self.layer2 = nn.Linear(128, 64) # Input: 128, Output: 64
        self.layer3 = nn.Linear(64, 10) # Input: 64, Output: 10 (digits 0-9)
    
    #Flatten image from [1,28,28] to [1, 784], reshaping into 1D vector
    def forward(self, x):
        x = x.view(-1, 28*28) 
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x) #No activation since loss function handles it
        return x
model = SimpleNN().to(device) #run with the gpu/cpu

#define loss function, optimizer
criterion = nn.CrossEntropyLoss() #Used for multi-class classification (logits -> probabilities)
optimizer = optim.Adam(model.parameters(), lr = 0.001) #Adaptive optimizer for faster convergence

#train model
num_epoch = 10
for epoch in range(num_epoch):
    running_loss = 0.0
    for image, label in train_loader:
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()  # Reset gradients
        outputs = model(image)  # Forward pass
        loss = criterion(outputs, label)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epoch}, Loss: {running_loss/len(train_loader):.4f}')

#eval model
correct = 0
total = 0
model.eval()  # Set model to evaluation mode

with torch.no_grad():  # Disable gradient calculations
    for image, label in test_loader:
        image, label = image.to(device), label.to(device)
        
        outputs = model(image)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get the class with highest score
        total += label.size(0)  # Total number of images
        correct += (predicted == label).sum().item()  # Count correct predictions
accuracy = 100 * correct / total
print(f"Accuracy of test: {accuracy:.2f}")

#visualize predictions
def visual():
    image, label = next(iter(test_loader))  # Get a batch of test images
    image, label = image[:10].to(device), label[:10].to(device)  # Take 10 images

    outputs = model(image)
    _, predictions = torch.max(outputs, 1)

    fig, axs = plt.subplots(1, 10, figsize=(12, 2))
    for i in range(10):
        ax = axs[i]
        ax.imshow(image[i].cpu().numpy().squeeze(), cmap='gray')
        ax.set_title(f"Prediction: {predictions[i].item()}")
    plt.show()

visual()