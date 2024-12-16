from abc import abstractmethod


class GeneratorModel:
    def __init__(self):
        self.prior_distribution_func = None

    def set_prior_distribution(self, func):
        """
        Sets the custom distribution function.

        Parameters:
            func (callable): A function that takes a sample size as input and returns samples.
        """
        if not callable(func):
            raise ValueError("The provided distribution must be a callable function.")
        self.prior_distribution_func = func

    def _sample(self, size):
        """
        Samples from the set distribution function.
        Parameters:
            size (int): Number of samples to generate.
        Returns:
            numpy.ndarray: Generated samples.
        """
        if self.prior_distribution_func is None:
            raise ValueError("No distribution function has been set.")
        return self.prior_distribution_func(size)

    @abstractmethod
    def _generate_latent_samples(self, nb_of_latent_samples: int):
        """
        Generate some samples using chose prior distribution.
        This will be used to feed the generator for inference.
        """
        pass

    @abstractmethod
    def generate_samples(self, nb_of_samples: int):
        """
        Generate some samples. This is the inference part.
        This is what we want to improve as the framework goal
        """
        pass

    @abstractmethod
    def update_generator(self, discriminator_predictions):
        """
        Child should calculate loss, then update itself accordingly here.
        The predictions are made on purely generated samples as inputs.
        A perfect prediction should yield 0.0 for all entries here.

        :param discriminator_predictions: predictions made by the discriminator.
        """
        pass

