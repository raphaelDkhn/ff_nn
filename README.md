# Stack Overflow Error on Cairo Projects

This project is a Cairo representation of a Neural Netwrork. 

We encounter the following error when we compile the project: 
```bash 
$ cd inference
$ scarb build
>>>
thread 'main' has overflowed its stack
fatal runtime error: stack overflow
zsh: abort      scarb build
```

Here is the Neural Network architecture: 
```python
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer
        self.fc3 = nn.Linear(64, 32)  # Third hidden layer
        self.fc4 = nn.Linear(32, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
```