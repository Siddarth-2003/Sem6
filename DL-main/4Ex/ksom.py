import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class KSOM(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate):
        super(KSOM, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.learning_rate = learning_rate
        self.sigma = output_dim / 2

        # Create the weight matrix
        self.weights = nn.Parameter(torch.randn(output_dim, output_dim, input_dim))

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, self.input_dim)

        # Compute the distance between each weight and the input
        distances = torch.sum((self.weights - x.unsqueeze(1)) ** 2, dim=-1)

        # Compute the winner neuron
        winner = torch.argmin(distances, dim=1)

        # Compute the neighborhood function
        neighborhood = torch.exp(-distances / (2 * self.sigma ** 2))

        # Update the weights
        delta = self.learning_rate * neighborhood.unsqueeze(-1) * (x.unsqueeze(1) - self.weights)
        self.weights.data += torch.sum(delta, dim=0)

        return winner

# Create some random input data
x_train = torch.randn(1000, 2)

# Create the KSOM network
ksom = KSOM(input_dim=2, output_dim=10, learning_rate=0.1)

# Train the network
for epoch in range(10):
    for i in range(len(x_train)):
        x = x_train[i]
        winner = ksom(x)

print(ksom.weights)

