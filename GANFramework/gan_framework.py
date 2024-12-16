import numpy as np
from GANFramework.discriminator_model import DiscriminatorModel
from GANFramework.generator_model import GeneratorModel
from data_utils.data_manipulation import DataManipulator


class GANFramework:

    def __init__(self, generator: GeneratorModel, discriminator: DiscriminatorModel, x_train, x_validation):
        self.generator = generator
        self.discriminator = discriminator
        self.x_train = x_train
        self.y_train = np.ones(x_train.shape[0])
        self.x_validation = x_validation
        self.y_validation = np.ones(x_validation.shape[0])

    def train(self, nb_of_epochs):
        """
        DISCRIMINATOR UPDATE STEP
        1.1 - Gather example from real dat
        1.2 - Label them all as 1 (aka real example)

        2.1 - Feed generator with sample from prior distribution to create samples
        2.2 - Label these with 0 (aka fake example)

        3.1 - concatenate fake and real samples, shuffle them

        4.1 - Train discriminator
        4.2 - update discriminator given the loss (D_step_per_epoch steps *usually 1*)

        GENERATOR UPDATE STEP
        5.1 - Feed generator with sample from prior distribution to create samples
        5.2 - Label these with 0 (aka fake example)

        6.1 - Make discriminator predict 0 or 1 on generated samples

        7.1 - Compute loss function of generator using discrminator predictions on generated samples
        7.2 - update generator given the loss (G_step_per_epoch steps *usually 1*)
        """

        discriminator_metrics_per_epoch = []

        # 1. Gather real examples
        real_samples_x = self.x_train
        real_samples_y = np.ones(self.y_train.shape[0])

        for epoch in range(nb_of_epochs):
            #########################
            #     Discriminator     #
            #########################

            # 2. Gather fake examples (using prior distribution)
            fake_samples_x = self.generator.generate_samples(nb_of_samples=real_samples_x.shape[0])
            fake_samples_y = np.zeros(real_samples_x.shape[0])

            # 3. Concatenate real and fake & shuffle
            mixed_x = np.vstack((real_samples_x, fake_samples_x))
            mixed_y = np.hstack((real_samples_y, fake_samples_y))

            indices = np.arange(len(mixed_x))
            np.random.shuffle(indices)
            mixed_x = mixed_x[indices]
            mixed_y = mixed_y[indices]

            # 4. Train discrminator for 1 epoch
            print(f"Training Discriminator for epoch {epoch} / {nb_of_epochs}")
            print(f"Fake: {fake_samples_x.shape[0]}, Real: {real_samples_x.shape[0]}")
            losses = self.discriminator.train(mixed_x, mixed_y, nb_of_epochs=50)
            print(f"Loss after training: {losses[-1]}")

            #########################
            #       Generator       #
            #########################

            # 5. Generate fake samples
            # *use same nb of samples as discriminator to be fair
            generated_samples = self.generator.generate_samples(mixed_x.shape[0])
            generated_samples_target_labels = np.ones(generated_samples.shape[0])

            # 6. Make discrminator predict if generator samples are real
            predictions_proba = self.discriminator.predict_proba(generated_samples)

            # 7. Update generator
            print(f"Training Generator for epoch {epoch} / {nb_of_epochs}")
            print(f"Fake: {generated_samples.shape[0]}")
            self.generator.update_generator(predictions_proba)

            # Evaluate discrminator
            report = self.evaluate_discriminator()
            print("Discriminator Evaluation Report for epoch " + str(epoch+1))
            print(f"Precision: {report['macro avg']['precision']}, Recall: {report['macro avg']['recall']}, F1: {report['macro avg']['f1-score']}")
            discriminator_metrics_per_epoch.append(report)

        # Return metrics #TODO: explore metric for generator evaluation
        return discriminator_metrics_per_epoch

    def evaluate_discriminator(self):
        real_samples_x = self.x_validation
        real_samples_y = np.ones(self.x_validation.shape[0])

        fake_samples_x = self.generator.generate_samples(nb_of_samples=real_samples_x.shape[0])
        fake_samples_y = np.zeros(fake_samples_x.shape[0])

        mixed_x = np.vstack((real_samples_x, fake_samples_x))
        mixed_y = np.hstack((real_samples_y, fake_samples_y))

        indices = np.arange(len(mixed_x))
        np.random.shuffle(indices)
        mixed_x = mixed_x[indices]
        mixed_y = mixed_y[indices]

        return self.discriminator.evaluate(mixed_x, mixed_y)

    def generate_samples(self, nb_of_samples):
        generated_samples = self.generator.generate_samples(nb_of_samples=nb_of_samples)
        return generated_samples



"""
This class is to search acceptable hyperparameter for the discriminator
before using it in the GAN Framework. Otherwise, GAN framework can find a 
low loss without having quality images
"""
class DiscriminatorFramework:

    def __init__(self, generator: GeneratorModel, discriminator: DiscriminatorModel, x_real_train, x_real_validation):
        self.generator = generator
        self.discriminator = discriminator

        # Track best model
        self.best_model = None
        self.best_model_idx = 0
        self.best_accuracy = 0

        # Build a dataset with fake examples
        self.x_train = x_real_train
        self.y_train = np.ones(x_real_train.shape[0])
        self.x_validation = x_real_validation
        self.y_validation = np.ones(x_real_validation.shape[0])

        # Training set
        fake_samples_train_x = self.generator.generate_samples(nb_of_samples=self.x_train.shape[0])
        fake_samples_train_y = np.zeros(fake_samples_train_x.shape[0])

        self.x_train = np.vstack((fake_samples_train_x, self.x_train))
        self.y_train = np.hstack((fake_samples_train_y, self.y_train))

        indices = np.arange(len(self.x_train))
        np.random.shuffle(indices)
        self.x_train  = self.x_train[indices]
        self.y_train = self.y_train[indices]

        # Validation set
        fake_samples_validation_x = self.generator.generate_samples(nb_of_samples=self.x_validation.shape[0])
        fake_samples_validation_y = np.zeros(fake_samples_validation_x.shape[0])

        self.x_validation = np.vstack((fake_samples_validation_x, self.x_validation))
        self.y_validation = np.hstack((fake_samples_validation_y, self.y_validation))

        indices = np.arange(len(self.x_validation))
        np.random.shuffle(indices)
        self.x_validation  = self.x_validation[indices]
        self.y_validation = self.y_validation[indices]

    def train(self, nb_of_epochs):
        discriminator_metrics_per_epoch = []
        loss_curve = []
        acc_curve = []


        print("Training Discriminator only")

        for epoch in range(nb_of_epochs):

            # 4. Train discriminator for 1 epoch
            print(f"Epoch {epoch} / {nb_of_epochs}")
            losses = self.discriminator.train(self.x_train, self.y_train, nb_of_epochs=1)
            # epoch += 4

            loss_curve.extend(losses)
            print(f"Loss after training: {losses[-1]}")

            # Evaluate discriminator
            report, accuracy = self.evaluate_discriminator()
            acc_curve.append(accuracy)

            if accuracy > self.best_accuracy or self.best_model is None:
                self.best_accuracy = accuracy
                self.best_model_idx = epoch
                self.best_model = self.discriminator

            print("Discriminator Evaluation Report for epoch " + str(epoch+1))
            print(f"Accuracy: {accuracy}, Precision: {report['macro avg']['precision']}, Recall: {report['macro avg']['recall']}, F1: {report['macro avg']['f1-score']}")
            discriminator_metrics_per_epoch.append(report)

        return discriminator_metrics_per_epoch, loss_curve, acc_curve

    def evaluate_discriminator(self):
        return self.discriminator.evaluate(self.x_validation, self.y_validation)

    def generate_samples(self, nb_of_samples):
        generated_samples = self.generator.generate_samples(nb_of_samples=nb_of_samples)
        return generated_samples