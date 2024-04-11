###### Your ID ######
# ID1: 123456789
# ID2: 987654321
#####################

# imports 
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
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################
    X = (X - np.mean(X)) / (np.max(X) - np.min(X))
    y = (y - np.mean(y)) / (np.max(y) - np.min(y))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.                             #
    ###########################################################################
    ones_arr = np.ones_like(X)
    X = np.column_stack((ones_arr,X))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    ###########################################################################
    m = X.shape[0]
    # calculate h(x) for each x into array
    h_x = np.sum((theta * X), axis=1)
    # print("X: ", X[0:3])
    # print("-------------------")
    # print("h(x): ", h_x[0:3])
    # print("-------------------")
    # print("y: ", y[0:3])
    # subtract each h(x) with y
    h_x -= y
    # print("subtract each h(x) with y: ", h_x[0:3])
    # square up: h(x)-y
    h_x *= h_x
    # print("square up: h(x)-y: ", h_x[0:3])
    J = (1/(2*m))*np.sum(h_x)
    # print("J: ", J)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    m = X.shape[0]
    for _ in range(num_iters): 
        J_history.append(compute_cost(X,y,theta))
        temp_theta_zero = theta[0] - alpha * (1/m) * (np.sum(np.sum((theta * X), axis=1)-y))
        temp_theta_one = theta[1] - alpha * (1/m) * (np.sum((np.sum((theta * X), axis=1)-y)*X[:,1]))
        theta = [temp_theta_zero,temp_theta_one]
    # print(theta)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    pinv_X = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X)
    pinv_theta = pinv_X @ y
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the efficient gradient descent optimization algorithm.  #
    ###########################################################################
    m = X.shape[0]
    for i in range(num_iters):
        J_history.append(compute_cost(X,y,theta))
        if len(J_history) > 1 and J_history[i-1] - J_history[i] < 1e-8:
            return theta
        temp_theta_zero = theta[0] - alpha * (1/m) * (np.sum(np.sum((theta * X), axis=1)-y))
        temp_theta_one = theta[1] - alpha * (1/m) * (np.sum((np.sum((theta * X), axis=1)-y)*X[:,1]))
        theta = [temp_theta_zero,temp_theta_one]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    theta = np.random.random(size=2)
    for a in alphas:
       theta_train = efficient_gradient_descent(X_train, y_train, theta, a, iterations)[0] 
       validation_loss = compute_cost(X_val, y_val, theta_train)
       alpha_dict[a] = validation_loss
    print(alpha_dict) 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    #####c######################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly

if __name__ == "__main__":
    arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

    print("Original array:")
    print(arr)
    # Sum along axis 0 (rows)
    row_sums = np.sum(arr, axis=0)
    print("Sum along axis 0 (rows):")
    print(row_sums)

    # Add your script's main functionality here
    # df = pd.read_csv('./data.csv')
    # X = df['sqft_living'].values
    # y = df['price'].values
    # X_train = [[1, 2, 3],
    #            [4, 5, 6],
    #            [7, 8, 9]]
    # y_train = [10, 20, 30]  
    # theta = np.array([-1, 2]) 
    # a = np.array((1,2,3))
    # b = np.array((2,3,4))
    # print(np.column_stack((a,b)))
    # X_train = apply_bias_trick(X)

    # print(X_train[:,1])
    # # compute_cost(X, y,theta)

