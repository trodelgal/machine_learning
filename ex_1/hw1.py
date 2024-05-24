import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """

    # Compute the maximum and minimum values of X
    X_max = np.max(X, axis = 0)
    X_min = np.min(X, axis = 0)
    # Compute the maximum and minimum values of y
    y_max = np.max(y, axis = 0)
    y_min = np.min(y, axis = 0)
    # Perform mean normalization
    X = (X - np.mean(X, axis = 0)) / (X_max - X_min)
    y = (y - np.mean(y, axis = 0)) / (y_max - y_min)
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """

    # Add a column of ones to the input data
    ones_arr = np.ones(X.shape[0])
    X = np.c_[(ones_arr,X)]
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    m = X.shape[0]
    h_x = X @ theta # Compute h(x)- predicted values
    J = (1/(2*m)) * np.sum(np.square(h_x - y)) # Compute the MSE
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    m = X.shape[0]
    for _ in range(num_iters): 
        h_theta_x = np.dot(X, theta) # Compute the predicted values
        X_tran = np.transpose(X) 
        error = h_theta_x - y # Compute the error
        theta = theta - alpha * np.dot(X_tran, error) / m  # Update theta 
        J_history.append(compute_cost(X, y, theta))  # Compute the cost and append it to the loss history
    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """  
    pinv_theta = []
    X_tran = X.T # Compute X transpose
    X_tran_mult_X = X_tran @ X # Compute X transpose mult X
    pinv_theta = np.linalg.inv(X_tran_mult_X).dot(X_tran)
    pinv_theta = np.dot(pinv_theta, y) # Compute pinv_theta transpose mult y
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    m = X.shape[0]
    for i in range(num_iters):
        h_theta_x = np.dot(X, theta)  # Compute predicted values
        X_tran = np.transpose(X) # Compute X transpose
        error = h_theta_x - y  # Compute the error
        theta = theta - alpha * np.dot(X_tran, error) / m # Update theta 
        J_history.append(compute_cost(X, y, theta)) # Compute the cost and append it to the loss history
        # checks improvement of the loss value is smaller than 1e-8
        if i > 1 and (J_history[i-1] - J_history[i]) < 1e-8:
            return theta, J_history
    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    # Initialize random theta
    np.random.seed(42)
    theta = np.random.random(X_train.shape[1])
    # Iterate through alpha values
    for a in alphas:
        # Train model using efficient gradient descent
       theta_train = efficient_gradient_descent(X_train, y_train, theta, a, iterations)[0] 
       validation_loss = compute_cost(X_val, y_val, theta_train)  # Compute validation loss
       alpha_dict[a] = validation_loss  # Store validation loss in dictionary
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """

    selected_features = []
     # Continue selecting features until we have 5
    while len(selected_features) < 5:
        features_dict={}
         # Iterate through each feature
        for i in range(X_train.shape[1]):
            if i not in selected_features:
                # Add feature to selected features 
                selected_features.append(i)
                # Apply bias trick to selected features
                selected_feature_train = apply_bias_trick(X_train[:,selected_features])
                selected_feature_val = apply_bias_trick(X_val[:,selected_features])
                  # Initialize random theta
                np.random.seed(42)
                theta = np.random.random(size = len(selected_features) + 1)
                # Train model using efficient gradient descent
                theta_train = efficient_gradient_descent(selected_feature_train, y_train, theta, best_alpha, iterations)[0] 
                 # Compute validation loss
                validation_loss = compute_cost(selected_feature_val, y_val, theta_train)
                features_dict[i] = validation_loss # Store validation loss for current feature
                selected_features.pop()  # Remove feature from selected features
        top_feature = min(features_dict, key=features_dict.get)  # Select feature with minimum validation loss
        selected_features.append(top_feature)
    
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    # feature_names = df_poly.columns
    # Initialize features dictionary
    features_dict={}
    # Iterate through each feature
    for i, feature_i in enumerate(df_poly.columns):
        # Add original feature to features dictionary
        features_dict[feature_i] = df_poly[feature_i]
        # Add squared feature to features dictionary
        features_dict[feature_i + "^2"] = df_poly[feature_i] * df_poly[feature_i]
        # Iterate through remaining features to create interaction features
        for feature_j in df_poly.columns[i+1:]:
            features_dict[feature_i + "*" + feature_j] = df_poly[feature_i] * df_poly[feature_j]
    # Create dataframe from features dictionary
    df_poly=pd.DataFrame(features_dict)
    return df_poly