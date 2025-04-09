import torch
import torchvision




import torch; print(torch.__version__)
import torchvision; print(torchvision.__version__)




import pandas as pd

# Load the CSV file
classes_csv_path = '/home/aarabambi/Meat Project/meat2/train/_classes.csv'  # Adjust the path if necessary
data = pd.read_csv(classes_csv_path)

# Display the first few rows of the dataframe
print(data.head())



import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        # Convert one-hot encoding to single label index
        row = self.img_labels.iloc[idx, 1:]
        label = row.idxmax()  # This will get the column name of the max value
        label_map = {' Fresh': 0, ' Half-Fresh': 1, ' Spoiled': 2}
        label_index = label_map[label]
        if self.transform:
            image = self.transform(image)
        return image, label_index




from torchvision import transforms

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create dataset instances
train_dataset = CustomImageDataset(
    annotations_file='/home/aarabambi/Meat Project/meat2/train/_classes.csv',
    img_dir='/home/aarabambi/Meat Project/meat2/train/images',  # Update this path based on where the images are stored
    transform=transform
)

valid_dataset = CustomImageDataset(
    annotations_file='/home/aarabambi/Meat Project/meat2/valid/_classes.csv',
    img_dir='/home/aarabambi/Meat Project/meat2/valid/images',  # Update this path based on where the images are stored
    transform=transform
)




from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=24)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=24)




import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean  # unnormalize
    plt.imshow(img.clip(0, 1))
    plt.show()

# Get a batch of training data
images, labels = next(iter(train_loader))

# Show images
fig = plt.figure(figsize=(25, 4))
for idx in range(20):
    ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(labels[idx].item())




import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Load a pre-trained ResNet and modify it for your number of classes
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # 3 classes

model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)




# train function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

# Optimizer is appropriate for CPU-based training
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)




device = torch.device("cpu")
model = model.to(device)  # Model is on the CPU




def validate(model, valid_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Validation Accuracy: {100 * correct / total}%')




train_model(model, train_loader, criterion, optimizer, num_epochs=10)
validate(model, valid_loader)





from sklearn.metrics import classification_report, confusion_matrix

def detailed_analysis(model, loader):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.view(-1).tolist())
            y_true.extend(labels.view(-1).tolist())
    
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Call the detailed analysis function for validation data
detailed_analysis(model, valid_loader)





torch.save(model.state_dict(), '/home/aarabambi/Meat Project/meat2/model_weights.pth')





model.load_state_dict(torch.load('/home/aarabambi/Meat Project/meat2/model_weights.pth'))
model.eval()  # Set to evaluation mode





torch.save(model.state_dict(), '/home/aarabambi/Meat Project/meat2/model_weights.gguf')







