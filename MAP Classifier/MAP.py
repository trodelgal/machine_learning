
import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.2,
            (1, 1): 0.5
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.045,
            (0, 0, 1): 0.045,
            (0, 1, 0): 0.105,
            (0, 1, 1): 0.105,
            (1, 0, 0): 0.105,
            (1, 0, 1): 0.105,
            (1, 1, 0): 0.245,
            (1, 1, 1): 0.245,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        for x, y in X_Y.keys():
            if np.all(np.isclose(X_Y[(x, y)], X[x] * Y[y])):
                return False  # X and Y are independent
        return True  # X and Y are dependent


    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        for x,y,c in X_Y_C.keys():
            if not np.isclose((X_Y_C[(x, y, c)] / C[c]), (X_C[(x, c)] / C[c]) * (Y_C[(y, c)] / C[c])):
                return False  # X and Y  given C are dependent
        return True  # X and Y are given C independent


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    rate_pow_k = rate ** k
    e_pow_minus_rate = np.exp(-rate)
    factorial_k = np.math.factorial(k)
    log_p = np.log2((rate_pow_k * e_pow_minus_rate) / factorial_k)
    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = np.zeros_like(rates)
    for i,rate in enumerate(rates):
        likelihood = 0
        for k in samples:
            likelihoods += poisson_log_pmf(k,rate)
        likelihoods[i] = likelihood  
    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help
    rate = rates[np.argmax(likelihoods)]
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    samples_sum = np.sum(samples)
    n = len(samples)
    mean = samples_sum / n
    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    # Calculate the exponent term
    exponent = -((x - mean)**2) / (2 * (std**2))
    coefficient = 1 / (np.sqrt(2 * np.pi * std**2))
    # Calculate the PDF
    p = coefficient * np.exp(exponent)
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        # Store the class value and dataset
        self.class_value = class_value
        self.data = np.copy(dataset)
        # Filter the dataset to only include instances with the specified class value
        class_data = dataset[dataset[:, -1] == class_value][:, :-1]
        # Compute the mean and standard deviation of the instances with the specified class value
        self.class_data = class_data
        self.mean = np.mean(class_data, axis = 0)
        self.std = np.std(class_data, axis = 0)
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        # Total number of samples in the dataset
        total_samples = len(self.data)  
        # Number of samples belonging to the class
        class_samples = len(self.class_data)  
        prior = class_samples / total_samples
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = 1
        # Iterate through each feature of the instance, except the last one (which is assumed to be the class label)
        for i in range(len(x)):
            # Multiply likelihood by the probability density function (PDF) of the feature given the mean and standard deviation
            # of the corresponding feature in the dataset distribution
           likelihood *= normal_pdf(x[i], self.mean[i], self.std[i])
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        # Get the likelihood probability of the instance under the class
        likelihood = self.get_instance_likelihood(x)
        # Get the prior probability of the class
        prior = self.get_prior()
        # Compute the posterior probability as the product of the likelihood and the prior
        posterior = likelihood * prior
        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        # Calculate the posterior probability for class 0 and class 1 using their respective conditional distributions
        prob_ccd0 = self.ccd0.get_instance_posterior(x)
        prob_ccd1 = self.ccd1.get_instance_posterior(x)

        # Compare the posterior probabilities to determine the predicted class
        if prob_ccd0 > prob_ccd1:
            pred = 0  # Predict class 0 if its posterior probability is higher
        else:
            pred = 1  # Predict class 1 otherwise

        # Return the predicted class
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    # Get the size of the test set
    test_set_size = len(test_set)
    # Initialize a counter for correctly classified samples
    correctly_classified = 0
    # Iterate through each sample in the test set
    for sample in test_set:
        # Check if the classifier predicts the same class label as the actual class label
        if map_classifier.predict(sample[:-1]) == sample[-1]:
            # If the prediction is correct, increment the counter
            correctly_classified += 1
    # Calculate the accuracy by dividing the number of correctly classified samples by the total number of samples
    acc = correctly_classified / test_set_size
    # Return the accuracy
    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    d = len(x)  # Number of dimensions/features
    det_cov = np.linalg.det(cov)  # Determinant of the covariance matrix
    inv_cov = np.linalg.inv(cov)  # Inverse of the covariance matrix

    # Calculate the exponent_term
    exponent_term = -0.5 * np.dot((x - mean).T, np.dot(inv_cov, (x - mean)))

    # Calculate the coefficient term
    coefficient_term = ((2 * np.pi) ** -(d / 2)) * (det_cov ** -0.5)

    # Calculate the probability density function (PDF)
    pdf = coefficient_term * np.exp(exponent_term)
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        # Store the class value and dataset
        self.class_value = class_value
        self.data = np.copy(dataset)
        # Filter the dataset to only include instances with the specified class value
        class_data = dataset[dataset[:, -1] == class_value][:, :-1]
        # Compute the mean and standard deviation of the instances with the specified class value
        self.class_data = class_data
        self.mean = np.mean(class_data, axis = 0)
        self.cov = np.cov(class_data.T)
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        # Total number of samples in the dataset
        total_samples = len(self.data)  
        # Number of samples belonging to the class
        class_samples = len(self.class_data)  
        prior = class_samples / total_samples
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        likelihood = multi_normal_pdf(x, self.mean, self.cov)
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        # Get the likelihood probability of the instance under the class
        likelihood = self.get_instance_likelihood(x)
        # Get the prior probability of the class
        prior = self.get_prior()
        # Compute the posterior probability as the product of the likelihood and the prior
        posterior = likelihood * prior
        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        # Calculate the posterior probability for class 0 and class 1 using their respective conditional distributions
        prior_ccd0 = self.ccd0.get_prior()
        prior_ccd1 = self.ccd1.get_prior()

        # Compare the prior probabilities to determine the predicted class
        if prior_ccd0 > prior_ccd1:
            pred = 0  # Predict class 0 if its prior probability is higher
        else:
            pred = 1  # Predict class 1 otherwise

        # Return the predicted class
        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        # Calculate the posterior probability for class 0 and class 1 using their respective conditional distributions
        likelihood_ccd0 = self.ccd0.get_instance_likelihood(x)
        likelihood_ccd1 = self.ccd1.get_instance_likelihood(x)

        # Compare the likelihood probabilities to determine the predicted class
        if likelihood_ccd0 > likelihood_ccd1:
            pred = 0  # Predict class 0 if its likelihood probability is higher
        else:
            pred = 1  # Predict class 1 otherwise

        # Return the predicted class
        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.class_value = class_value
        self.data = np.copy(dataset)
        # Filter the dataset to only include instances with the specified class value
        self.class_data = dataset[dataset[:, -1] == class_value][:, :-1]
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = None
        # Total number of samples in the dataset
        total_samples = len(self.data)  
        # Number of samples belonging to the class
        class_samples = len(self.class_data)  
        prior = class_samples / total_samples
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
         # Initialize likelihood to 1
        likelihood = 1.0 
        # Number of instances in the class
        n_i = len(self.class_data)
        for i in range(len(x)):
            n_ij = np.sum(self.class_data[:, i] == x[i]) # Count occurrences of x[i] in the i-th feature column
            V_j = len(np.unique(self.class_data[:, i])) # Number of unique values in the i-th feature column
            if n_ij == 0 and np.sum(self.data[:, i] == x[i]) == 0:
                likelihood *= EPSILLON
            else:
                # Calculate the likelihood using Laplace smoothing
                likelihood *= np.prod((n_ij + 1) / (n_i + V_j))
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        # Get the likelihood probability of the instance under the class
        likelihood = self.get_instance_likelihood(x)
        # Get the prior probability of the class
        prior = self.get_prior()
        # Compute the posterior probability as the product of the likelihood and the prior
        posterior = likelihood * prior
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        # Calculate the posterior probability for class 0 and class 1 using their respective conditional distributions
        prob_ccd0 = self.ccd0.get_instance_posterior(x)
        prob_ccd1 = self.ccd1.get_instance_posterior(x)

        # Compare the posterior probabilities to determine the predicted class
        if prob_ccd0 > prob_ccd1:
            pred = 0  # Predict class 0 if its posterior probability is higher
        else:
            pred = 1  # Predict class 1 otherwise

        # Return the predicted class
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        # Get the size of the test set
        test_set_size = len(test_set)
        # Initialize a counter for correctly classified samples
        correctly_classified = 0
        # Iterate through each sample in the test set
        for sample in test_set:
            # Check if the classifier predicts the same class label as the actual class label
            if self.predict(sample[:-1]) == sample[-1]:
                # If the prediction is correct, increment the counter
                correctly_classified += 1
        # Calculate the accuracy by dividing the number of correctly classified samples by the total number of samples
        acc = correctly_classified / test_set_size
        # Return the accuracy
        return acc


