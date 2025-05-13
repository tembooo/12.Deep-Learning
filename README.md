# 12.Deep-Learning
 Target Center Estimation Using Deep Learning
🧠 Target Center Estimation Using Deep Learning
This project focuses on estimating the center coordinates of a target from input images using neural networks. Designed as a hands-on workshop, it guides you through the stages of image data onboarding, optional feature extraction, and splitting datasets into training, test, and validation subsets. You’ll implement and adapt a deep learning model (using PyTorch, JAX, or other frameworks) for regression-based coordinate prediction. The project emphasizes training evaluation and visualization of predicted vs. actual center points. It’s ideal for learners aiming to apply CNNs or custom architectures to real-world visual regression problems.

![image](https://github.com/user-attachments/assets/df5bb2ed-7752-49a4-a67e-baff4b8d3aac)

🎯 Typed Content:
The goal of this workshop is to solve the following task:
Given an image of a target, estimate the coordinates of its center.
![target image]

![image](https://github.com/user-attachments/assets/21ebb62b-deca-43f5-bf09-14b0326e1d0c)

Carry out the following tasks as needed by implementing the necessary code for MATLAB or Python:
Data onboarding: load and understand the data.
Feature extraction (optional): eventually the task is to statistically classify the samples based on some features. Consider whether feature extraction would be needed to properly classify the samples. If necessary, extract a new feature or features from the existing ones.
Data division: the data is already divided into train and test subsets. When training neural networks, it is a good idea to also use a validation subset to estimate the generalization during the training.
Implementing the training procedure: it is allowed to use higher level frameworks for deep learning (e.g. Pytorch, JAX, etc). In order to get familiar with the framework, it would be a good idea to start with an example provided by the framework authors. The next step would be to modify it to work with the provided data (Is the network architecture suitable? Should the loss function be changed?)
Evaluate the trained network on the test subset: calculate the loss function using the test subset. Is it comparable to the loss on the training subset? On the validation subset? Visualize the results by plotting an image, true center and predicted center for some samples. How far off is the predicted value?
The dataset: ZIP
The coordinates are specified assuming that (0, 0) is in the top-left corner of an image.

```python
import os
                                                                                    # extracted data 
ArmanGolbidi_image_directory = 'images'                                             # images directory
ArmanGolbidi_training_csv_path = 'train.csv'                                        # train.csv path
ArmanGolbidi_testing_csv_path = 'test.csv'                                          # test.csv path
                                                                                    # verify
print(f"Image directory: {ArmanGolbidi_image_directory}")
print(f"Training CSV path: {ArmanGolbidi_training_csv_path}")
print(f"Testing CSV path: {ArmanGolbidi_testing_csv_path}")
                                                                                    # Check
if not os.path.exists(ArmanGolbidi_image_directory):
    print(f"Error: Image directory does not exist at {ArmanGolbidi_image_directory}")
if not os.path.exists(ArmanGolbidi_training_csv_path):
    print(f"Error: Training CSV does not exist at {ArmanGolbidi_training_csv_path}")
if not os.path.exists(ArmanGolbidi_testing_csv_path):
    print(f"Error: Testing CSV does not exist at {ArmanGolbidi_testing_csv_path}")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
ArmanGolbidi_training_dataframe = pd.read_csv(ArmanGolbidi_training_csv_path)
ArmanGolbidi_testing_dataframe = pd.read_csv(ArmanGolbidi_testing_csv_path)
class ArmanGolbidiDataset(Dataset):
    def __init__(self, dataframe, directory_path, image_transform=None):
        self.dataframe = dataframe
        self.directory_path = directory_path
        self.image_transform = image_transform

    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, index):
        image_file_name = os.path.join(self.directory_path, self.dataframe.iloc[index, 0])
        image = Image.open(image_file_name).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        coordinates = torch.tensor([self.dataframe.iloc[index, 1], self.dataframe.iloc[index, 2]], dtype=torch.float32)
        return image, coordinates
# Combine all image transformations into a pipeline.
ArmanGolbidi_image_transformations = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
ArmanGolbidi_training_dataset = ArmanGolbidiDataset(ArmanGolbidi_training_dataframe, ArmanGolbidi_image_directory, image_transform=ArmanGolbidi_image_transformations)
ArmanGolbidi_testing_dataset = ArmanGolbidiDataset(ArmanGolbidi_testing_dataframe, ArmanGolbidi_image_directory, image_transform=ArmanGolbidi_image_transformations)
ArmanGolbidi_training_size = int(0.8 * len(ArmanGolbidi_training_dataset))
ArmanGolbidi_validation_size = len(ArmanGolbidi_training_dataset) - ArmanGolbidi_training_size
ArmanGolbidi_training_dataset, ArmanGolbidi_validation_dataset = random_split(ArmanGolbidi_training_dataset, [ArmanGolbidi_training_size, ArmanGolbidi_validation_size])

ArmanGolbidi_training_loader = DataLoader(ArmanGolbidi_training_dataset, batch_size=32, shuffle=True)
ArmanGolbidi_validation_loader = DataLoader(ArmanGolbidi_validation_dataset, batch_size=32, shuffle=False)
ArmanGolbidi_testing_loader = DataLoader(ArmanGolbidi_testing_dataset, batch_size=32, shuffle=False)

class ArmanGolbidiCNN(nn.Module):
    def __init__(self):
        super(ArmanGolbidiCNN, self).__init__()
        self.convolution1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.convolution2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.convolution3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fully_connected1 = nn.Linear(64 * 16 * 16, 128)
        self.fully_connected2 = nn.Linear(128, 2)
        self.pooling_layer = nn.MaxPool2d(2, 2)
        self.activation = nn.ReLU()
    def forward(self, x):
        x = self.pooling_layer(self.activation(self.convolution1(x)))
        x = self.pooling_layer(self.activation(self.convolution2(x)))
        x = self.pooling_layer(self.activation(self.convolution3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.activation(self.fully_connected1(x))
        x = self.fully_connected2(x)
        return x
ArmanGolbidi_neural_network = ArmanGolbidiCNN()
ArmanGolbidi_loss_function = nn.MSELoss()  # Consider if MSE is appropriate for coordinate prediction
ArmanGolbidi_optimizer_function = optim.Adam(ArmanGolbidi_neural_network.parameters(), lr=0.001)
ArmanGolbidi_total_epochs = 32
for epoch in range(ArmanGolbidi_total_epochs):
    ArmanGolbidi_neural_network.train()
    ArmanGolbidi_running_loss_value = 0.0
    for images, coordinates in ArmanGolbidi_training_loader:
        ArmanGolbidi_optimizer_function.zero_grad()
        outputs = ArmanGolbidi_neural_network(images)
        loss = ArmanGolbidi_loss_function(outputs, coordinates)
        loss.backward()
        ArmanGolbidi_optimizer_function.step()
        ArmanGolbidi_running_loss_value += loss.item()
    ArmanGolbidi_neural_network.eval()
    ArmanGolbidi_validation_loss_value = 0.0
    with torch.no_grad():
        for images, coordinates in ArmanGolbidi_validation_loader:
            outputs = ArmanGolbidi_neural_network(images)
            loss = ArmanGolbidi_loss_function(outputs, coordinates)
            ArmanGolbidi_validation_loss_value += loss.item()
    print(f"Epoch [{epoch+1}/{ArmanGolbidi_total_epochs}], Training Loss: {ArmanGolbidi_running_loss_value/len(ArmanGolbidi_training_loader):.4f}, Validation Loss: {ArmanGolbidi_validation_loss_value/len(ArmanGolbidi_validation_loader):.4f}")

ArmanGolbidi_neural_network.eval()
ArmanGolbidi_testing_loss_value = 0.0
with torch.no_grad():
    for images, coordinates in ArmanGolbidi_testing_loader:
        outputs = ArmanGolbidi_neural_network(images)
        loss = ArmanGolbidi_loss_function(outputs, coordinates)
        ArmanGolbidi_testing_loss_value += loss.item()

print(f"Test Loss: {ArmanGolbidi_testing_loss_value/len(ArmanGolbidi_testing_loader):.4f}")

# %%
import matplotlib.pyplot as plt

# Visualize predictions on test data
ArmanGolbidi_neural_network.eval()
with torch.no_grad():
    for i, (images, coordinates) in enumerate(ArmanGolbidi_testing_loader):
        if i == 2:  # Visualize first 2 batches
            break
        outputs = ArmanGolbidi_neural_network(images)

        for j in range(len(images)):
            plt.imshow(images[j].permute(1, 2, 0))
            true_x, true_y = coordinates[j].numpy()
            pred_x, pred_y = outputs[j].numpy()
            plt.scatter([true_x], [true_y], color='red', label='Real Center')
            plt.scatter([pred_x], [pred_y], color='black', label='Center of prediction')
            plt.legend()
            plt.show()

```
![image](https://github.com/user-attachments/assets/f29e3985-cc4f-4ac0-8dbc-85d967699c03)
