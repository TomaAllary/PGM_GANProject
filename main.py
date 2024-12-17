import datetime
import os

import numpy as np
from matplotlib import pyplot as plt

from GANFramework.cnn_discriminator_model import CNNDiscriminatorModel
from GANFramework.gan_framework import GANFramework, DiscriminatorFramework
from GANFramework.generator_model import GeneratorModel
from GANFramework.logistic_regression_discriminator_model import LogisticRegressionDiscriminatorModel
from GANFramework.mlp_discriminator_model import MLPDiscriminatorModel
from GANFramework.pytorch_nn_generator_model import PyTorchNNGeneratorModel
from GANFramework.svm_discriminator_model import SVMDiscriminatorModel
from data_utils.data_manipulation import DataManipulator
from data_utils.mnist_loader import MNISTLoader


# from ExperiencesUtils.ModelTrainer import ModelTrainer
# def model_trainer_example(x_train, y_train):
#     clf = MLPClassifierModel(np.reshape(x_train, (x_train.shape[0], -1)), y_train)
#
#     trainer = ModelTrainer(clf)
#
#     # 2x2x3=12 combinations
#     possibilities = {
#         "solver": ["sgd"],
#         "validation_fraction": [0.1],
#         "learning_rate": ["constant", "adaptive"],
#         "learning_rate_init": [0.01, 0.1],
#         "regularization_rate": [0.001],
#         "layers": [(3,), (500, 300, 100), (5, 9, 3)],
#     }
#     trainer.set_hyperparameters_possibilities(possibilities)
#     trainer.grid_search()


def generate_images(generator: GeneratorModel, nb_of_images=10):

    folder_id = "generation_"
    date_now = datetime.datetime.now()
    folder_id += date_now.strftime("%m")
    folder_id += '_'
    folder_id += date_now.strftime("%y")
    folder_id += '_'
    folder_id += date_now.strftime("%d")
    folder_id += '-'
    folder_id += date_now.strftime("%H")
    folder_id += date_now.strftime("%M")
    folder_id += date_now.strftime("%S")

    data_dir_path = os.path.join(os.path.dirname(__file__), "data", folder_id)
    if not os.path.exists(data_dir_path):
        os.mkdir(data_dir_path)

    width = 4
    length = int(nb_of_images / width) + 1

    generated_images = generator.generate_samples(nb_of_images)

    i = 0
    for img in generated_images:
        re_formatted = img * 255

        ax = plt.subplot(length, width, i + 1)
        ax.imshow(re_formatted.squeeze(), cmap="gray")  # Assuming grayscale images
        ax.axis("off")  # Turn off axis for better visualization

        i += 1

    # Save the entire figure
    plt.tight_layout()  # Adjust layout to prevent overlap
    path = os.path.join(data_dir_path, f'generated_images.png')
    plt.savefig(path, bbox_inches="tight")  # Save the plot as an image
    plt.close()  # Close the figure to free memory


def discriminator_grid_search(data_manipulator: DataManipulator):

    generator_model = PyTorchNNGeneratorModel(100, (28, 28))

    # "learning_rate": 0.01,
    # "regularization_rate": 0.001,
    # "batch_size": 32,
    for batch_size in [3000, ]:
        for lr in [1.0, 0.01, 1e-2, 1e-3, 1e-4]:
            discriminator_model = MLPDiscriminatorModel(image_shape=(28, 28)) #Best yet: lr=0.001 reg=0.0 batch=3000
            # discriminator_model = CNNDiscriminatorModel() #Best yet: lr=0.001 reg=0.0 batch=3000

            discriminator_model.set_hyperparameter('learning_rate', lr)
            discriminator_model.set_hyperparameter('batch_size', batch_size)
            discriminator_model.set_hyperparameter('regularization_rate', 0.0)

            train_discriminator_only(data_manipulator, generator_model, discriminator_model, param_str=f"Lr={lr}, reg=0.0, batch_size={batch_size}")

def train_discriminator_only(data_manipulator: DataManipulator, generator, discriminator, param_str=None):

    discr = DiscriminatorFramework(generator, discriminator, data_manipulator.x_train, data_manipulator.x_validation)

    discriminator_metrics_per_epoch, loss_curve, acc_curve = discr.train(20)

    folder_id = "discriminator_"
    date_now = datetime.datetime.now()
    folder_id += date_now.strftime("%m")
    folder_id += '_'
    folder_id += date_now.strftime("%y")
    folder_id += '_'
    folder_id += date_now.strftime("%d")
    folder_id += '-'
    folder_id += date_now.strftime("%H")
    folder_id += date_now.strftime("%M")
    folder_id += date_now.strftime("%S")

    data_dir_path = os.path.join(os.path.dirname(__file__), "data", folder_id)
    if not os.path.exists(data_dir_path):
        os.mkdir(data_dir_path)

    additional_title = ""
    if param_str:
        additional_title = f"({param_str})"

    # LOSS
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_curve) + 1), loss_curve)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss per Epoch {additional_title}")

    # Save the plot
    plt.tight_layout()  # Adjust layout to prevent overlap
    path = os.path.join(data_dir_path, f'loss_curve.png')
    plt.savefig(path, bbox_inches="tight")  # Save the plot as an image
    plt.close()  # Close the figure to free memory

    # ACCURACY
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(acc_curve) + 1), acc_curve)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Validation Accuracy per Epoch {additional_title}")

    # Save the plot
    plt.tight_layout()  # Adjust layout to prevent overlap
    path = os.path.join(data_dir_path, f'acc_curve.png')
    plt.savefig(path, bbox_inches="tight")  # Save the plot as an image
    plt.close()  # Close the figure to free memory

def main():
    # Retrieve mnist data (28x28 images with normalized pixel value between 0 and 1)
    loader = MNISTLoader()
    loader.set_normalize_image()
    loader.set_specific_digit(0)
    x_train, y_train, x_test, y_test = loader.load_mnist_data()

    # Handle imbalance (Not necessary for mnist)
    data_manipulator = DataManipulator(x_train, y_train)
    # data_manipulator.show_class_distribution()
    # dataManipulator.apply_oversampling_train_set()
    # dataManipulator.show_class_distribution()
    data_manipulator.create_validation_set(validation_ratio=0.2)

    # GAN FRAMEWORK
    generator_model = PyTorchNNGeneratorModel(100, (28,28))
    generator_model.set_hyperparameter('learning_rate', 0.001)
    generator_model.set_hyperparameter('regularization_rate', 0.0)

    # This is MLP with best param (found with training discriminator alone)
    discriminator_model = MLPDiscriminatorModel(image_shape=(28,28))
    discriminator_model.set_hyperparameter('learning_rate', 0.001)
    discriminator_model.set_hyperparameter('batch_size', 3000)
    discriminator_model.set_hyperparameter('regularization_rate', 0.0)


    # This is CNN with best param (found with training discriminator alone)
    # discriminator_model = CNNDiscriminatorModel()
    # discriminator_model.set_hyperparameter('learning_rate', 0.001)
    # discriminator_model.set_hyperparameter('batch_size', 3000)
    # discriminator_model.set_hyperparameter('regularization_rate', 0.0)


    framework = GANFramework(
        generator=generator_model,
        discriminator=discriminator_model,
        x_train=data_manipulator.x_train,
        x_validation=data_manipulator.x_validation
    )

    framework.train(55)
    # Show generation model images
    generate_images(framework.generator, 12)

    # Train Discriminator only
    # train_discriminator_only(data_manipulator, generator_model, discriminator_model, param_str="SKLearn Logistic regression lbfs")

    # Grid Search
    # discriminator_grid_search(data_manipulator)

if __name__=="__main__":
    main()