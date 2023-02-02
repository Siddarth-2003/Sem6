import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model and loss function
model = Net()
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(10000):
    # Input data
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    # True labels
    y_true = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

    # Forward pass
    y_pred = model(x)
    loss = criterion(y_pred, y_true)

    # Backward pass and weight update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Check the model's prediction
x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
y_pred = model(x)
print(y_pred)
