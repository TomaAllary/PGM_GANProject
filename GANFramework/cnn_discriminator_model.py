import numpy as np

from GANFramework.discriminator_model import DiscriminatorModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class CNNModel(nn.Module):
    def __init__(self, ):

        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # 128 channels, 3x3 image size after pooling
        self.fc2 = nn.Linear(256, 1)  # Output one value for binary classification

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(-1, 128 * 3 * 3)  # Flatten the tensor for the fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        x = self.sigmoid(x)  # Apply sigmoid to get output between 0 and 1 (binary classification)
        return x


class CNNDiscriminatorModel(DiscriminatorModel):
    def __init__(self, ):
        self.hyperparameters = self.__default_hyperparameters()
        device_used = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"MLPDiscriminatorModel is using {device_used}")
        self.device = torch.device(device_used)

        # Define the model
        self.cnn_model = CNNModel().to(self.device)

        # Define loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = None

    # Override parent function
    def should_flatten_inputs(self):
        return False

    # HYPERPARAMETERS
    def __default_hyperparameters(self) -> dict:
        dict = {
            "learning_rate":0.01,
            "regularization_rate":0.001,
            "batch_size": 32,
        }
        return dict

    def set_hyperparameter(self, hyperparam_name: str, hyperparam_value):
        if not self.hyperparameters.__contains__(hyperparam_name):
            print("Trying to set unknown hyper param")
            return

        if type(self.hyperparameters[hyperparam_name]) != type(hyperparam_value):
            print(f"Trying to set hyper param of type {type(self.hyperparameters[hyperparam_name])} with a value of type {type(hyperparam_value)}")
            return

        self.hyperparameters[hyperparam_name] = hyperparam_value

    #Define @abstractmethod
    def train(self, inputs, labels, nb_of_epochs=1):

        if self.optimizer is None:
            self.optimizer = optim.Adam(
                self.cnn_model.parameters(),
                lr=self.hyperparameters["learning_rate"],
                weight_decay=self.hyperparameters["regularization_rate"]
            )

        inputs = np.expand_dims(inputs, axis=1)  # Adding channel dimension, result shape: (50000, 1, 28, 28)

        # Create DataLoader
        dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.hyperparameters["batch_size"], shuffle=True)

        epochs_losses = []

        # Training loop
        for epoch in range(nb_of_epochs):
            self.cnn_model.train() # set to 'training state'
            epoch_loss = 0.0
            for batch_inputs, batch_labels in dataloader:
                batch_inputs, batch_labels = batch_inputs.to(self.device), batch_labels.to(self.device)

                self.optimizer.zero_grad() # Reset gradient
                outputs = self.cnn_model(batch_inputs)
                loss = self.criterion(outputs.squeeze(), batch_labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epochs_losses.append(epoch_loss)

        return epochs_losses

    #Define @abstractmethod
    def predict_proba(self, inputs):
        """
        Predict probability of inputs being real
        """
        self.cnn_model.eval() # Set to 'evaluation state'
        with torch.no_grad():
            inputs = np.expand_dims(inputs, axis=1)  # Adding channel dimension, result shape: (50000, 1, 28, 28)
            inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            probabilities = self.cnn_model(inputs)

        return probabilities.squeeze().cpu().numpy()

