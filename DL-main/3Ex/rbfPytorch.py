import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
X, y = load_iris(return_X_y=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define RBF layer
class RBF(nn.Module):
    def __init__(self, num_centers, in_features, out_features):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, in_features))
        self.widths = nn.Parameter(torch.randn(num_centers))
        self.linear = nn.Linear(num_centers, out_features)
    
    def radial(self, X):
        dist = torch.sum((X[:, None, :] - self.centers[None, :, :]) ** 2, dim=2)
        return torch.exp(-dist / (2 * self.widths ** 2))
    
    def forward(self, X):
        radial_output = self.radial(X)
        return self.linear(radial_output)

# Define RBF network
class RBFNetwork(nn.Module):
    def __init__(self, num_centers, in_features, out_features):
        super().__init__()
        self.rbf = RBF(num_centers, in_features, out_features)
    
    def forward(self, X):
        return self.rbf(X)

# Define training loop
def train(model, X, y, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Define model hyperparameters
num_centers = 10
in_features = X_train.shape[1]
out_features = 1
learning_rate = 0.1
num_epochs = 1000

# Initialize model, optimizer, and loss function
model = RBFNetwork(num_centers, in_features, out_features)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Train model
train(model, torch.tensor(X_train).float(), torch.tensor(y_train).float(), optimizer, criterion, num_epochs)

# Evaluate model on test set
with torch.no_grad():
    y_pred = model(torch.tensor(X_test).float())
    y_pred_class = (y_pred > 0.5).float()
    accuracy = torch.mean((y_pred_class == torch.tensor(y_test).float()).float())
    print(f"Test Accuracy: {accuracy.item():.4f}")
