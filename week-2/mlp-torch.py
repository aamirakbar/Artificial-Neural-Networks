import torch
import torch.nn as nn

# Define the model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim1)
        self.hidden_layer1 = nn.Linear(hidden_dim1, hidden_dim2)
        self.hidden_layer2 = nn.Linear(hidden_dim2, output_dim)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        x = self.input_layer(inputs)
        x = self.activation(x)
        x = self.hidden_layer1(x)
        x = self.activation(x)
        x = self.hidden_layer2(x)
        x = torch.tanh(x)
        return x

# Create an instance of the model
model = MLP(input_dim=5, hidden_dim1=7, hidden_dim2=7, output_dim=1)

# Print model summary
print(model)
