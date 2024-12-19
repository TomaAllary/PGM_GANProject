import datetime
import os

import numpy as np
from matplotlib import pyplot as plt

from GANFramework.cnn_discriminator_model import CNNDiscriminatorModel
from GANFramework.gan_framework import GANFramework, DiscriminatorFramework
from GANFramework.generator_model import GeneratorModel
from GANFramework.mlp_discriminator_model import MLPDiscriminatorModel
from GANFramework.pytorch_nn_generator_model import PyTorchNNGeneratorModel
from data_utils.data_manipulation import DataManipulator
from data_utils.mnist_loader import MNISTLoader

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

    print(generated_images[0].max())
    print(generated_images[0].min())

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
    # Search parameter that will distinguish the most a real from fake/noise

    # Declare a fixed generator (untrained in this case)
    generator_model = PyTorchNNGeneratorModel(100, (28, 28))

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

    return discr.discriminator

def generator_grid_search(data_manipulator: DataManipulator):
    # Search parameter that will fool the most a discriminator

    # MLP Discriminator with best param (found with training discriminator alone)
    discriminator_model = MLPDiscriminatorModel(image_shape=(28,28))
    discriminator_model.set_hyperparameter('learning_rate', 0.001)
    discriminator_model.set_hyperparameter('batch_size', 3000)
    discriminator_model.set_hyperparameter('regularization_rate', 0.1)


    versions = ["v2", None] #v2=cnn, None=nn
    latent_input_size = 100
    learning_rates = [0.01]#, 0.001, 0.0001, 0.1, 1.0]

    for version in versions:
        for learning_rate in learning_rates:
            generator_model = PyTorchNNGeneratorModel(latent_size=latent_input_size, image_shape=(28, 28), version=version)

            framework = GANFramework(
                generator=generator_model,
                discriminator=discriminator_model,
                x_train=data_manipulator.x_train,
                x_validation=data_manipulator.x_validation
            )

            discriminator_metrics_per_epoch = framework.train(50, fix_discriminator=False)

            acc_curve = [entry['accuracy'] for entry in discriminator_metrics_per_epoch]


            ######################################################
            # plot
            #####################################################
            v = version
            if v is None:
                v = "v1"
            additional_title = f"Gen {v}, learning rate: {learning_rate}"

            folder_id = "generator_"
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

            # Discriminator ACCURACY
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(acc_curve) + 1), acc_curve)
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"Discriminator Accuracy per Epoch {additional_title}")
            plt.tight_layout()  # Adjust layout to prevent overlap
            path = os.path.join(data_dir_path, f'discriminator_acc_curve.png')
            plt.savefig(path, bbox_inches="tight")  # Save the plot as an image
            plt.close()  # Close the figure to free memory

            generate_images(framework.generator, 12)

def main():
    #################################
    #     Pre-Process MNIST data    #
    #################################
    # Retrieve mnist data (28x28 images with normalized pixel value between 0 and 1)
    loader = MNISTLoader()
    loader.set_normalize_image()
    # loader.set_specific_digit(0)
    x_train, y_train, x_test, y_test = loader.load_mnist_data()

    # Handle imbalance (Not necessary for mnist)
    data_manipulator = DataManipulator(x_train, y_train)
    # data_manipulator.show_class_distribution()
    # dataManipulator.apply_oversampling_train_set()
    # dataManipulator.show_class_distribution()
    data_manipulator.create_validation_set(validation_ratio=0.2)

    #################################
    #     Define G & D models       #
    #################################
    # NN Generator
    generator_model = PyTorchNNGeneratorModel(100, (28,28))
    generator_model.set_hyperparameter('learning_rate', 0.001)
    generator_model.set_hyperparameter('regularization_rate', 0.0)

    # MLP Discriminator with best param (found with training discriminator alone)
    discriminator_model = MLPDiscriminatorModel(image_shape=(28,28))
    discriminator_model.set_hyperparameter('learning_rate', 0.001)
    discriminator_model.set_hyperparameter('batch_size', 3000)
    discriminator_model.set_hyperparameter('regularization_rate', 0.1)
    # Less optimal lr, to slow it down (similar to training the generator more often)
    # discriminator_model.set_hyperparameter('learning_rate', 0.0001)

    # CNN Discriminator with best param (found with training discriminator alone)
    # discriminator_model = CNNDiscriminatorModel()
    # discriminator_model.set_hyperparameter('learning_rate', 0.001)
    # discriminator_model.set_hyperparameter('batch_size', 3000)
    # discriminator_model.set_hyperparameter('regularization_rate', 0.0)


    #################################
    #       Train GAN Framework     #
    #################################
    # framework = GANFramework(
    #     generator=generator_model,
    #     discriminator=discriminator_model,
    #     x_train=data_manipulator.x_train,
    #     x_validation=data_manipulator.x_validation
    # )

    # framework.train(55)
    # Show generation model images
    # generate_images(framework.generator, 12)


    #################################
    #   Train Discriminator only    #
    #################################
    # Train Discriminator only
    # discriminator_model = train_discriminator_only(data_manipulator, generator_model, discriminator_model, param_str="SKLearn Logistic regression lbfs")


    ###############################################################
    #       Grid Search for either Discriminator or Generator     #
    ###############################################################
    # Grid Search
    # discriminator_grid_search(data_manipulator)
    generator_grid_search(data_manipulator)



if __name__=="__main__":
    main()



