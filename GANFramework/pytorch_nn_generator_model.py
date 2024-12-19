from GANFramework.generator_model import GeneratorModel

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class PyTorchNNModel(torch.nn.Module):
    def __init__(self, input_size, output_shape):
        super().__init__()
        self.output_shape = output_shape
        last_layer_size = output_shape[0] * output_shape[1]

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, last_layer_size),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, self.output_shape[0], self.output_shape[1])
        return output

class PyTorchNNModel_v2(torch.nn.Module):
    def __init__(self, input_size, output_shape):
        super().__init__()
        self.output_shape = output_shape
        last_layer_size = output_shape[0] * output_shape[1]

        self.model = torch.nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.3),
            nn.Linear(2048, last_layer_size),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, self.output_shape[0], self.output_shape[1])
        return output

class PyTorchNNGeneratorModel(GeneratorModel):
    def __init__(self, latent_size, image_shape, version=None):
        super().__init__()
        self.hyperparameters = self.__default_hyperparameters()

        device_used = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"PyTorchNNGeneratorModel is using {device_used}")
        self.device = torch.device(device_used)

        if version is None:
            self.nn_model = PyTorchNNModel(latent_size, image_shape)
        if version == "v2":
            self.nn_model = PyTorchNNModel_v2(latent_size, image_shape)

        self.loss_function = torch.nn.BCELoss()
        self.optimizer = None

        self.latent_input_size = 100

        # Set prior for latent space
        def gaussian_distribution(size, mean=0, std=1):
            return np.random.normal(loc=mean, scale=std, size=size)
        self.set_prior_distribution(lambda size: gaussian_distribution(size, mean=5, std=2))

    # HYPERPARAMETERS
    def __default_hyperparameters(self) -> dict:
        dict = {
            "learning_rate":0.001,
            "regularization_rate":0.001,
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
    def _generate_latent_samples(self, nb_of_latent_samples: int):
        """
        Generate some samples using chose prior distribution.
        This will be used to feed the generator for inference.
        """
        latent_samples = []
        for i in range(nb_of_latent_samples):
            latent_samples.append(self._sample(self.latent_input_size))

        return np.array(latent_samples)

    #Define @abstractmethod
    def generate_samples(self, nb_of_samples: int):
        """
        Generate some samples. This is the inference part.
        This is what we want to improve as the framework goal

        Return: array of ( nb_of_samples X self.latent_input_size )
        """
        latent_samples = self._generate_latent_samples(nb_of_samples)
        latent_samples_tensor = torch.tensor(latent_samples, dtype=torch.float32).to(self.device)

        print(latent_samples_tensor.shape)
        logits = self.nn_model(latent_samples_tensor) # tensor of shape (n, 1, 28, 28)
        generated_samples = torch.sigmoid(logits)

        generated_samples = generated_samples.squeeze(1)  # Remove the dimension at index 1

        # convert to np and remap [-1, 1] -> [0, 1]
        # return ((generated_samples.cpu().detach().numpy()) + 1) / 2

        return (generated_samples.cpu().detach().numpy())


    #Define @abstractmethod
    def update_generator(self, discriminator_predictions):
        """
        Child should calculate loss, then update itself accordingly here.
        The predictions are made on purely generated samples as inputs.
        A perfect prediction should yield 0.0 for all entries here.

        :param discriminator_predictions: predictions probability made by the discriminator. Np array of (n,)
        """
        self.nn_model.zero_grad()

        if self.optimizer is None:
            self.optimizer = optim.Adam(
                self.nn_model.parameters(),
                lr=self.hyperparameters["learning_rate"],
                weight_decay=self.hyperparameters["regularization_rate"]
            )

        print(f"Count of missclassified images for generator update: {discriminator_predictions[(discriminator_predictions > 0.5)].shape[0]}")

        # Build tensors
        discriminator_predictions_tensor = torch.tensor(
            discriminator_predictions,
            dtype=torch.float32,
            requires_grad=True,
            device=self.device
        )
        # Have REAL (=1) labels to compute loss
        target_labels = torch.ones(discriminator_predictions.shape[0], device=self.device)

        # Compute loss where generator goal is to have the discriminator predict REAL(=1) for all samples
        loss = self.loss_function(discriminator_predictions_tensor.squeeze(), target_labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()
