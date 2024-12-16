from abc import abstractmethod

import numpy as np
from sklearn.metrics import classification_report, accuracy_score


class DiscriminatorModel:

    def should_flatten_inputs(self):
        return False

    @abstractmethod
    def train(self, inputs, labels, nb_of_epochs=1):
        pass

    @abstractmethod
    def predict_proba(self, inputs):
        """
        Predict probability of inputs being real
        """
        pass

    def predict(self, inputs):
        """
        Predict the most probably class (0 or 1) (fake or real)
        """
        predictions_proba = self.predict_proba(inputs)
        predictions = np.argmax(predictions_proba, axis=1)

        return predictions

    def evaluate(self, inputs, labels):
        predictions = self.predict(inputs)

        acc = accuracy_score(labels, predictions)
        return classification_report(labels, predictions, output_dict=True), acc

