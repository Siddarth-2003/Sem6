import torch
import torch.nn as nn

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

# Initialize the model and loss function
model = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    # Input data
    x = torch.tensor([[1.0]])
    # True label
    y_true = torch.tensor([[17.0]])

    # Forward pass
    y_pred = model(x)
    loss = criterion(y_pred, y_true)

    # Backward pass and weight update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Check the model's prediction
x = torch.tensor([[1.0]])
y_pred = model(x)
print(y_pred)
