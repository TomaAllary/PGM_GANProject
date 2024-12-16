import numpy as np
from matplotlib import pyplot as plt


class DataManipulator:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.x_validation = None
        self.y_validation = None

    def create_validation_set(self, validation_ratio=0.2):
        validation_size = int(len(self.x_train) * validation_ratio)

        self.x_validation = self.x_train[:validation_size]
        self.y_validation = self.y_train[:validation_size]
        self.x_train = self.x_train[validation_size:]
        self.y_train = self.y_train[validation_size:]

    def show_class_distribution(self):
        unique_values, counts = np.unique(self.y_train, return_counts=True)
        # bar chart
        plt.bar(unique_values, counts, color='blue', edgecolor='black', alpha=0.7)
        plt.title("Training set - Classes Counts")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.show()

        if self.y_validation is not None:
            unique_values, counts = np.unique(self.y_validation, return_counts=True)
            # bar chart
            plt.bar(unique_values, counts, color='blue', edgecolor='black', alpha=0.7)
            plt.title("Validation set - Classes Counts")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.show()

    def shuffle_data(self):
        indices = np.arange(len(self.x_train))
        np.random.shuffle(indices)

        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]

    def apply_undersampling(self, x, y):
        """
        Applies undersampling to balance the classes in self.y_train.
        This method modifies x and y to ensure that
        all classes have the same number of samples as the minority class.
        """
        unique_values, counts = np.unique(y, return_counts=True)
        lowest_count = np.min(counts)

        selected_indices = []
        for label in unique_values:
            class_indices = np.where(y == label)[0]

            selected_indices.extend(np.random.choice(class_indices, lowest_count, replace=False)) # Randomly add indices

        # Shuffle selected indices to maintain randomness
        np.random.shuffle(selected_indices)

        # Update the training set
        x = x[selected_indices]
        y = y[selected_indices]
        return x, y

    def apply_undersampling_train_set(self):
        self.x_train, self.y_train = self.apply_undersampling(self.x_train, self.y_train)
    def apply_undersampling_validation_set(self):
        self.x_validation, self.y_validation = self.apply_undersampling(self.x_validation, self.y_validation)

    def apply_oversampling(self, x, y):
        """
        Applies oversampling to balance the classes in self.y_train.
        This method modifies self.x_train and self.y_train to ensure that
        all classes have the same number of samples as the majority class.
        """
        unique_values, counts = np.unique(y, return_counts=True)
        highest_count = np.max(counts)

        oversampled_indices = []
        for label in unique_values:
            class_indices = np.where(y == label)[0]

            oversampled_indices.extend(
                np.random.choice(class_indices, highest_count, replace=True)
            )  # Randomly add indices

        # Shuffle selected indices to maintain randomness
        np.random.shuffle(oversampled_indices)

        # Update the training set
        x = x[oversampled_indices]
        y = y[oversampled_indices]

        return x, y

    def apply_oversampling_train_set(self):
        self.x_train, self.y_train = self.apply_oversampling(self.x_train, self.y_train)
    def apply_oversampling_validation_set(self):
        self.x_validation, self.y_validation = self.apply_oversampling(self.x_validation, self.y_validation)


