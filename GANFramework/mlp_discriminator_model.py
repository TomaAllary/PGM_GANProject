from GANFramework.discriminator_model import DiscriminatorModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MLPModel(nn.Module):
    def __init__(self, input_size):
        self.input_size = input_size
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), self.input_size)
        output = self.model(x)
        return output


class MLPDiscriminatorModel(DiscriminatorModel):

    def __init__(self, image_shape):
        self.hyperparameters = self.__default_hyperparameters()
        device_used = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"MLPDiscriminatorModel is using {device_used}")
        self.device = torch.device(device_used)

        # Define the model
        self.mlp_model = MLPModel(input_size=(image_shape[0] * image_shape[1])).to(self.device)

        # Define loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = None

    # Override parent function
    def should_flatten_inputs(self):
        return True

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
                self.mlp_model.parameters(),
                lr=self.hyperparameters["learning_rate"],
                weight_decay=self.hyperparameters["regularization_rate"]
            )

        # Create DataLoader
        dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.hyperparameters.get("batch_size", 32), shuffle=True)

        epochs_losses = []

        # Training loop
        for epoch in range(nb_of_epochs):
            self.mlp_model.train() # set to 'training state'
            epoch_loss = 0.0
            for batch_inputs, batch_labels in dataloader:
                batch_inputs, batch_labels = batch_inputs.to(self.device), batch_labels.to(self.device)

                self.optimizer.zero_grad() # Reset gradient
                outputs = self.mlp_model(batch_inputs)
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
        self.mlp_model.eval() # Set to 'evaluation state'
        with torch.no_grad():
            inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            outputs = self.mlp_model(inputs)
            probabilities = torch.softmax(outputs, dim=1)

        return probabilities.cpu().numpy()

