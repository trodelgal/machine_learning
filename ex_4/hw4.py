

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def pearson_correlation( x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    r = None
    # Calculate the mean of the two columns
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    # Calculate the Pearson correlation coefficient
    x_minus_mean_x = x - mean_x
    y_minus_mean_y = y - mean_y
    numerator = np.sum(x_minus_mean_x * y_minus_mean_y)

    denominator = np.sqrt(np.sum(x_minus_mean_x ** 2) * np.sum(y_minus_mean_y ** 2))
    #Check for homogeneous columns
    if np.sum(x_minus_mean_x ** 2) and np.sum(y_minus_mean_y ** 2) == 0:
        r = np.nan
    elif denominator == 0:
        r = 0
    else:
        r = numerator / denominator
    return r

def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    # Convert X to a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Drop columns "id" and "date" if they exist
    X = X.drop(columns=[col for col in ['id', 'date'] if col in X], errors='ignore')

    # Compute Pearson correlation for each column
    p_correlations = X.apply(lambda col: pearson_correlation(col, y))

    # Select top n_features based on the absolute value of the correlation coefficient
    best_features = p_correlations.abs().sort_values(ascending=False).head(n_features).index.tolist()

    return best_features

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []


    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # Apply the bias trick
        X = self.apply_bias_trick(X)
        # set random seed
        np.random.seed(self.random_state)
        # Initialize theta with random values
        self.theta = np.random.random(X.shape[1])
        number_of_samples = X.shape[0]
        # Perform gradient descent
        for _ in range(self.n_iter):
          # Calculate the predicted probabilities
          #type of theta_x is vactor of size m
          theta_x = np.dot(X, self.theta)
          #type of h_theta is vactor of size m
          sigmoid_theta_x = self.sigmoid(theta_x)
          # Calculate the gradient
          gradient = np.dot(X.T, (sigmoid_theta_x - y))
          # Update theta
          self.theta -= self.eta * (gradient/number_of_samples)
          self.thetas.append(self.theta)
          # Calculate the cost
          cost = self.culc_cost(X, y)
          #cost = (np.sum(-y * np.log(sigmoid_theta_x) - (1 - y) * np.log(1 - sigmoid_theta_x)))/number_of_samples
          self.Js.append(cost)
          # Check for convergence
          if len(self.Js) > 1 and abs(self.Js[-1] - self.Js[-2]) < self.eps:
            break
       

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        # Apply the bias trick
        X = self.apply_bias_trick(X)
        sigmoid_theta_x = self.sigmoid(np.dot(X, self.theta))
        # Convert the probabilities to class labels
        preds = [1 if x >= 0.5 else 0 for x in sigmoid_theta_x]
        return np.array(preds)
    
    def sigmoid(self, theta_x):
        return 1.0 / (1.0 + np.exp(-theta_x))
       
    def apply_bias_trick(self,X):
      """
      Applies the bias trick to the input data.

      Input:
      - Y: Input data (m instances over n features).

      Returns:
      - X: Input data with an additional column of ones in the
          zeroth position (m instances over n+1 features).
      """
      # Add a column of ones to the input data
      ones_arr = np.ones(X.shape[0])
      X = np.c_[(ones_arr,X)]
      return X
    

    def culc_cost(self, X, y):
      """
      Calculate the cost for the given data.

      Input:
      - X: Input data (m instances over n features).
      - y: True labels (m instances).

      Returns:
      - cost: The cost associated with the current set of parameters.
      """
      theta_x = np.dot(X, self.theta)
      sigmoid_theta_x = self.sigmoid(theta_x)
      # Calculate the cost
      cost = (-1.0 / len(y)) * (np.dot(y.T, np.log(sigmoid_theta_x)) + np.dot((1 - y).T, np.log(1 - sigmoid_theta_x)))
      return cost
  
   

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """
    cv_accuracy = None
    # set random seed
    np.random.seed(random_state)
    # Shuffle the data
    shuffled_indices = np.random.permutation(len(y))
    X = X[shuffled_indices]
    y = y[shuffled_indices]
    # Split the data to folds
    X_folds = np.array_split(X, folds)
    y_folds = np.array_split(y, folds)
    # Perform cross validation
    accuracies = []
    for i in range(folds):
        # Split the data to train and test - i is the test fold
        X_train = np.concatenate([X_folds[j] for j in range(folds) if j != i])
        y_train = np.concatenate([y_folds[j] for j in range(folds) if j != i])
        X_test = X_folds[i]
        y_test = y_folds[i]
        # Train the model
        algo.fit(X_train, y_train)
        # Test the model
        preds = algo.predict(X_test)
        # Calculate the accuracy
        accuracy = np.sum(preds == y_test)/len(y_test)
        accuracies.append(accuracy)
    cv_accuracy = np.mean(accuracies)
    return cv_accuracy



def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    # Calculate the exponent term
    exponent = -((data - mu)**2) / (2 * (sigma**2))
    # Calculate the coefficient term
    coefficient = 1.0 / (np.sqrt(2 * np.pi * sigma**2))
    # Calculate the PDF
    p = coefficient * np.exp(exponent)
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        n_samples  = data.shape[0]
        # Initialize weights to be uniform
        self.weights = np.full(self.k, 1.0 / self.k)
        # Initialize responsibilities to be an array of size  matrice of n_sumple x k filled with zeros
        self.responsibilities = np.zeros((n_samples, self.k))
        # Randomly choose k samples as initial cluster centers
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.mus = data[random_indices].reshape(self.k)
        # Initialize sigmas to be random
        self.sigmas = np.random.uniform(0, 1, self.k)
        # Initialize the cost array
        self.costs = []

    def expectation(self, data):
      """
      E step - This function should calculate and update the responsibilities
      """
       # Calculate the numerator of the responsibilities
      numinator = self.weights * norm_pdf(data, self.mus.T, self.sigmas) 
      # Sum across components for each data point
      sum_pdf = np.sum(numinator, axis=1, keepdims=True)
      # Normalize responsibilities so that they sum to 1 for each data point
      self.responsibilities = numinator / sum_pdf
        
    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        # Update the weights
        self.weights = np.mean(self.responsibilities, axis=0)
        # Update mus
        self.mus = np.dot(self.responsibilities.T, data).reshape(self.k) / (self.weights * len(data))
        #self.mus = np.sum(self.responsibilities * data.reshape(-1, 1), axis=0) / np.sum(self.responsibilities, axis=0)
        # Update the sigmas
        variance = np.mean(self.responsibilities * np.square(data.reshape(-1, 1) - self.mus), axis=0)
        self.sigmas = np.sqrt(variance / self.weights)
        
    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        # Initialize the distribution params
        self.init_params(data)
        
        for _ in range(self.n_iter):
            # Perform the Expectation step
            self.expectation(data)
            # Perform the Maximization step
            self.maximization(data)
            # Calculate the cost
            cost = self.calc_cost(data)
            self.costs.append(cost)
            # Check for convergence
            if len(self.costs) > 1 and abs(self.costs[-1] - self.costs[-2]) < self.eps:
                break

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas


    def calc_cost(self, data):
      """
      Calculate the cost for the given data.
      
      Input:  
      - data: Input data (m instances over n features).
      
      Returns:
      - cost: The cost associated with the current set of parameters."""
      cost = 0
      # Calculate the cost
      for row in data:
          cost_j = [-1 * np.log(norm_pdf(row, self.mus[k], self.sigmas[k]) * self.weights[k]) for k in range(self.k)]
          cost += np.sum(cost_j)
      return cost
    

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    # Calculate the pdf
    pdf = None
    pdf = np.sum(weights * norm_pdf(data.reshape(-1,1), mus, sigmas), axis=1)
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.priors = None
        self.class_params = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        np.random.seed(self.random_state)
        self.priors = {}
        self.class_params = {}
        # Get the unique classes
        classes = np.unique(y)
        # Train a GMM for each class
        for class_value in classes:
            # Calculate the prior
            self.priors[class_value] = np.mean(y == class_value)
            # Get the data for the current class
            class_data = X[y == class_value]
            feature_models = []
            for feature in range(X.shape[1]):
                feature_col = class_data[:, feature].reshape(-1, 1)
                # Train the EM
                feature_em = EM(k=self.k, random_state=self.random_state)
                feature_em.fit(feature_col)
                feature_models.append(feature_em.get_dist_params())
            self.class_params[class_value] = feature_models

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []
        for instance in X:
            posteriors = [(self.priors[class_label] * self.calc_likelihood(instance,class_label),class_label) for class_label in self.priors.keys()]
            max_prob = max(posteriors, key=lambda t: t[0])
            # Append the predicted class label to the list of predictions
            preds.append(max_prob[1])  
        return np.array(preds)
    
    def calc_likelihood(self, x, class_value):
      """
      Calculate the likelihood of a given instance.

      Input:
      - x: A given instance.
      - class_value: The class value.

      Returns:
      - likelihood: The likelihood of the instance.
      """

      likelihood = 1
      for feature in range(x.shape[0]):
          weights, mus, sigmas = self.class_params[class_value][feature]
          gmm = gmm_pdf(x[feature], weights, mus, sigmas)
          likelihood = likelihood * gmm
      return likelihood
    

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    # Initialize the models
    logistic_regression_model = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    naive_bayes_gaussian_model = NaiveBayesGaussian(k=k)

    # Train the models
    logistic_regression_model.fit(x_train, y_train)
    naive_bayes_gaussian_model.fit(x_train, y_train)

    # Test the models
    lor_train_preds = logistic_regression_model.predict(x_train)
    lor_test_preds = logistic_regression_model.predict(x_test)
    bayes_train_preds = naive_bayes_gaussian_model.predict(x_train)
    bayes_test_preds = naive_bayes_gaussian_model.predict(x_test)
    
    # Calculate the accuracies
    lor_train_acc = np.sum(lor_train_preds == y_train) / len(y_train)
    lor_test_acc = np.sum(lor_test_preds == y_test) / len(y_test)
    bayes_train_acc = np.sum(bayes_train_preds == y_train) / len(y_train)
    bayes_test_acc = np.sum(bayes_test_preds == y_test) / len(y_test)

    # generate plot titles
    if len(y_train) == 1000:
        naive_plot_title = "Naive Bayes - first 1,000 training instances"
        lor_plot_title = "Logistic Regression - first 1,000 training instances"
    else:
        naive_plot_title = "Naive Bayes - all training instances"
        lor_plot_title = "Logistic Regression - all training instances"

    # Plot the decision boundaries
    plot_decision_regions(x_train, y_train, naive_bayes_gaussian_model, title = naive_plot_title)
    print("Accuracy: ", bayes_train_acc)
    if len(y_train) == 1000:
      print("Main observation: The Naive Bayes model shows strong performance, clearly illustrating that each class originates from a distinct Gaussian distribution.")
    else:
      print("Main observation: Great performance of the Naive Bayes classifier model. It recognizes and separates the classes effectively.")
    plot_decision_regions(x_train, y_train, logistic_regression_model , title = lor_plot_title)
    print("Accuracy: ", lor_train_acc)
    if len(y_train) == 1000:
      print("Main observation: high performance for the Logistic Regression model, The data is almost linearly separable.")
    else:
      print("Main observation: Poor performance of the Logistic Regression model. Since the data isn't linearly separable, the algorithm doesn't recognize and separate the two classes effectively.")

    plot_lor_cost_function(logistic_regression_model)
    if len(y_train) == 1000:
      print("Main observation: There is a nice cost improvement along the way. however, since the data isn't perfectly linearly separable, the algorithm didn't reach zero cost.")
    else:
      print("Main observation: The learning algorithm stopped very early because no better classifier was found.")
   
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    np.random.seed(0)
      
    # Generate dataset_a
    # Generate 3 Gaoussian distributions without linear separability and low covariance
    mean_a1 = [4.5, 4.5, 4.5]
    cov_a1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    mean_a2 = [10, 10, 10]
    cov_a2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    mean_a3 = [0.5, 0.5, 0.5]
    cov_a3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    size_a1 = 100
    size_a2 = 50
    size_a3 = 50
    
    class_a1 = multivariate_normal.rvs(mean=mean_a1, cov=cov_a1, size=size_a1)
    class_a2 = multivariate_normal.rvs(mean=mean_a2, cov=cov_a2, size=size_a2)
    class_a3 = multivariate_normal.rvs(mean=mean_a3, cov=cov_a3, size=size_a3)
    
    dataset_a_features = np.vstack((class_a1, class_a2,class_a3))
    dataset_a_labels = np.hstack((np.zeros(size_a1), np.ones(size_a1)))
    
    # Generate dataset_b
    # Generate 2 Gaoussian distributions with linear separability and high covariance
    mean_b1 = [1, 2, 5]
    cov_b1 = [[1, 0.9, 0.9], [0.9, 1, 0.9], [0.9, 0.9, 1]]
    
    mean_b2 = [1, 2, 3]
    cov_b2 = [[1, 0.9, 0.9], [0.9, 1, 0.9], [0.9, 0.9, 1]]
    
    size_b = 100
    
    class_b1 = multivariate_normal.rvs(mean=mean_b1, cov=cov_b1, size=size_b)
    class_b2 = multivariate_normal.rvs(mean=mean_b2, cov=cov_b2, size=size_b)
    
    dataset_b_features = np.vstack((class_b1, class_b2))
    dataset_b_labels = np.hstack((np.zeros(size_b), np.ones(size_b)))
    
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }

# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
    plt.show()

# Function for ploting the cost function of the logistic regression model
def plot_lor_cost_function(lor):
    plt.plot(range(1, len(lor.Js) + 1), lor.Js)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Logistic Regression - Cost vs Iteration')
    plt.show()


