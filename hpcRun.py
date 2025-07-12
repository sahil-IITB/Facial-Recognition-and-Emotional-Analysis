# Importing Necessary Libraries

import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


#########################################################################################################

# Model Creation
class EmotionAnalyser(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # ==> Output Size [32, 48, 48]
        self.bn1 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2,2) # Output Size ==> [32, 24, 24]
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Output Size [64, 24, 24] --> [64, 12, 12]
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Output Size [128, 12, 12] --> [128, 6, 6]
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # Output Size [256, 6, 6] --> [256, 3, 3]
        self.bn4 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*3*3, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, xb):
        out = self.pool(self.relu(self.bn1(self.conv1(xb))))
        out = self.pool(self.relu(self.bn2(self.conv2(out))))
        out = self.pool(self.relu(self.bn3(self.conv3(out))))
        out = self.pool(self.relu(self.bn4(self.conv4(out))))
        out = self.dropout2(out)
        out = out.view(-1, 256 * 3 * 3)
        out = self.relu(self.fc1(out))
        out = self.dropout1(out)
        out = self.fc2(out)

        return out
    
#########################################################################################################

# Helper Functions to load data to GPU if available
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    '''Wrap a dataloader to move data to device'''
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        '''Yield a batch of data after moving it to device'''
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        '''Number of batches'''
        return len(self.dl)

#########################################################################################################

# Variables, Parameters, Paths

# Creating Folder to save the plots
save_dir = "plots"
os.makedirs(save_dir, exist_ok=True)

# Dataset Path
csv_path = r"/scratch/IITB/vikram-gadre-r-and-d-group/24m1087/models/Facial/dataset/fer2013.csv"

# Checking whether GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-Parameters
learning_rate = 1e-3
epochs = 50

# Defining Model
model = EmotionAnalyser()
to_device(model, device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)



#########################################################################################################

# Dataset Loading
class FerDataset(torch.utils.data.Dataset):
    def __init__(self, ds_path, split='Training', transform=None):
        self.ds = pd.read_csv(ds_path)
        self.ds = self.ds[self.ds['Usage'] == split] 
        self.transform = transform

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        label = int(self.ds.iloc[index]['emotion'])
        pixels = np.array(self.ds.iloc[index]['pixels'].split(), dtype='uint8')
        image = pixels.reshape(48, 48) 
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label
    

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_loader = DataLoader(FerDataset(csv_path, split='Training', transform=train_transform), batch_size=64, shuffle=True)
val_loader = DataLoader(FerDataset(csv_path, split='PublicTest', transform=val_test_transform), batch_size=64)
test_loader = DataLoader(FerDataset(csv_path, split='PrivateTest', transform=val_test_transform), batch_size=64)
        
#########################################################################################################

# Loss Calculation and Weigt Updation During Training
def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    preds = model(xb)
    loss = loss_func(preds, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result

# Evaluation funtion to evaluate model metric on validation and test dataset
def evaluate(model, loss_fn, valid_dl, metric=None):
    with torch.no_grad():
        results = [loss_batch(model, loss_fn, xb, yb, metric=metric) for xb, yb in valid_dl]
        losses, nums, metrics = zip(*results)
        total_size = np.sum(nums)
        avg_loss = np.sum(np.multiply(losses,nums)) / total_size
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums)) / total_size

        return avg_loss, total_size, avg_metric
    
# Training function
def fit(epochs, model, loss_fn, opt, train_dl, valid_dl, metric=None):
    best_accuracy = 0.0

    accuracies, losses = [], []
    for epoch in range(epochs):
        for xb, yb in train_dl:
            loss, _, _ = loss_batch(model, loss_fn, xb, yb, opt)

        result = evaluate(model, loss_fn, valid_dl, metric)
        val_loss, total, val_metric = result
        scheduler.step(val_loss)

        if val_metric > best_accuracy:
            best_accuracy = val_metric
            torch.save(model.state_dict(), 'best_model.pth')  # Save best model
            print(f"Saved best model at epoch {epoch+1} with accuracy {val_metric:.4f}")

        if metric is None:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, val_loss))
        else:
            print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'.format(epoch+1, epochs, val_loss, metric.__name__, val_metric))
            accuracies.append(val_metric)
        
        losses.append(val_loss)

    return accuracies, losses

# Metric
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)

#########################################################################################################

# Training the model
accuracies, losses = fit(epochs, model, F.cross_entropy, optimizer, train_loader, val_loader, accuracy)

#########################################################################################################

# Plotting loss and accuracy graphs
epochs_range = range(1, len(accuracies) + 1)

plt.figure(figsize=(12, 5))

# Accuracy subplot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracies, '-o', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. No. of Epochs')

# Loss subplot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, losses, '-o', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. No. of Epochs')

plt.tight_layout()
plot_path = os.path.join(save_dir, f"train_metrics_lr{learning_rate}_ep{epochs}.png")
plt.savefig(plot_path, dpi=300)
plt.show()

#########################################################################################################

# Accuracy on test dataset
test_loader = DeviceDataLoader(test_loader, device)

test_loss, total, test_acc = evaluate(model, F.cross_entropy, test_loader, metric=accuracy)
print('Loss: {:4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))

#########################################################################################################

# Collect predictions and true labels
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Emotion labels
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

plt.title("FER-2013 Confusion Matrix (Normalized)", fontsize=16)
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
filename = f"conf_matrix_lr{str(learning_rate).replace('.', '_')}_ep{epochs}.png"
save_path = os.path.join(save_dir, filename)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()