import torch
import torch.nn as nn

# Define a custom module for the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(2, 1)
    def forward(self, x):
        x = self.hidden(x)
        return torch.sigmoid(x)

# Instantiate the network
net = Net()

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# Prepare the input and target data
x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Train the network
for epoch in range(1000):
    optimizer.zero_grad()
    output = net(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# Print the final output of the network
print(output)
