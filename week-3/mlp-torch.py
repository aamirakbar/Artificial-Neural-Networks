import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

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

# Create an instance of the model
model = MLP(input_dim=5, hidden_dim1=7, hidden_dim2=7, output_dim=1)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

# Train the model
epochs = 50
batch_size = 32
for epoch in range(epochs):
    for batch_start in range(0, len(X_train_tensor), batch_size):
        batch_end = batch_start + batch_size
        X_batch = X_train_tensor[batch_start:batch_end]
        y_batch = y_train_tensor[batch_start:batch_end]

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(X_batch)  # Forward pass
        loss = criterion(outputs, y_batch)  # Calculate the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the parameters

# Evaluate the model on test data
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    test_accuracy = ((torch.round(torch.sigmoid(test_outputs)) == y_test_tensor).sum().item()) / len(y_test_tensor)
    print("Test Loss:", test_loss.item())
    print("Test Accuracy:", test_accuracy)
