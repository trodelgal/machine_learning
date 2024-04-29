###### Your ID ######
# ID1: 322529447
# ID2: 206133597
#####################

import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    # Get unique labels and their counts
    unique_values, counts = np.unique(data[:,-1],return_counts=True)
    # Calculate the total number of instances in the dataset
    dataset_size = data.shape[0]
    # Calculate the squared proportions of each label
    counts = np.square(counts/dataset_size)
    # Calculate Gini impurity as 1 - sum of squared proportions
    gini = 1.0 - np.sum(counts)
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    # Get unique labels and their counts
    unique_values, counts = np.unique(data[:,-1],return_counts=True)
    # Calculate the total number of instances in the dataset
    dataset_size = data.shape[0]
    # Calculate the proportions of each label and compute the logarithm
    proportions = counts / dataset_size
    log_probs = np.log2(proportions)
    # Calculate entropy as the negative sum of proportion times logarithm
    entropy = -np.sum(proportions * log_probs)
    return entropy

class DecisionNode:

    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        # Get unique values and their counts for the Node feature
        unique_values, counts = np.unique(self.data[:,self.feature],return_counts=True)
        # Find the index of the most frequent value
        max_index = np.argmax(counts)
        # Set the prediction to the most frequent value
        pred = unique_values[max_index]
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        # Append the child node to the list of children
        self.children.append(node)
        # Append the value associated with the child node to the list of children values       
        self.children_values.append(val)
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        unique_values, counts = np.unique(self.data[:,self.feature],return_counts=True)
        unique_values_impurity = []
        # Calculate impurity for each unique value of the selected feature
        for val in unique_values:
            unique_values_impurity.append(self.impurity_func(self.data[self.data[:,self.feature] == val]))
        # Calculate feature importance
        self.feature_importance = (len(self.data)/n_total_sample) * self.impurity_func(self.data) - np.sum((counts/n_total_sample) * unique_values_impurity)
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        # Get unique feature values and their counts
        unique_values, counts = np.unique(self.data[:,feature],return_counts=True)
        # Total size of the dataset
        dataset_size = len(self.data)
        # List to store impurity values for each unique feature value
        unique_values_impurity = []
        # Split the dataset based on each unique feature value
        for val in unique_values:
            groups[val] = self.data[self.data[:,feature] == val]
            # Calculate impurity for the subset based on the impurity function
            if self.gain_ratio:
                unique_values_impurity.append(calc_entropy(groups[val]))
            else:
                unique_values_impurity.append(self.impurity_func(groups[val]))
        # Calculate goodness of split based on impurity values
        if self.gain_ratio:
            # Calculate information gain and split information
            information_gain = calc_entropy(self.data) - np.sum((counts/dataset_size) * unique_values_impurity)
            split_information = -np.sum((counts/dataset_size) * np.log2(counts/dataset_size))
            if split_information != 0:  # Avoid division by zero
                goodness = information_gain / split_information
            else:
                goodness = 0.0  # Set goodness to 0 if split_information is 0
        else:
            # For other impurity functions, simply subtract the impurity of subsets from overall impurity
            goodness = self.impurity_func(self.data) - np.sum((counts/dataset_size) * unique_values_impurity)
        return goodness, groups
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        # Check if the node has reached the maximum depth
        if self.max_depth <= self.depth:
            self.terminal = True
            return
        
        # Initialize variables to keep track of the best feature and its goodness
        best_feature_index = None
        best_feature_goodnes = 0.0
        best_feature_groups = {}

        # Iterate over each feature to find the best split
        for i in range(self.data.shape[1] - 1):
            goodness, split_values = self.goodness_of_split(i)
            if goodness > best_feature_goodnes:
                best_feature_index = i
                best_feature_goodnes = goodness
                best_feature_groups = split_values

        # If no meaningful split is found, mark the node as terminal
        if len(best_feature_groups) <= 1 or best_feature_goodnes == 0.0:
            self.terminal = True
            return
        
        # Prune the node based on chi-square test if chi value is set
        if self.chi != 1:
            unique_values, counts = np.unique(self.data[:,-1],return_counts=True)
            chi_square = calc_chi_square(best_feature_groups,unique_values, counts)
            freedom_degree = (len(best_feature_groups) - 1) * (len(unique_values) - 1)
            if chi_square < chi_table[freedom_degree][self.chi]:
                self.terminal = True
                return
            
        # Set the best feature for splitting
        self.feature = best_feature_index

        # Create child nodes for each split group
        for feature, data_subset in best_feature_groups.items():
            child_node = DecisionNode(data = data_subset, impurity_func = self.impurity_func, feature = -1, depth = self.depth + 1, chi = self.chi, max_depth = self.max_depth, gain_ratio = self.gain_ratio)
            self.add_child(child_node,feature)

                    
class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        # Create the root node of the tree
        self.root = DecisionNode(data = self.data, impurity_func = self.impurity_func, feature=-1,depth=0, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
        # save the total sample to calculate the FI
        n_total_sample = len(self.data)
        # Initialize a queue for breadth-first traversal        
        queue = [self.root]

        # Loop until the queue is empty
        while queue:
            # Get the current node from the queue
            current_node = queue.pop(0)
            # Split the current node to create children nodes
            current_node.split()
            # calculate the FI
            current_node.feature_importance = current_node.calc_feature_importance(n_total_sample)
            # Add the children nodes to the queue for further processing
            queue.extend(current_node.children or [])



    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        node = self.root
        while not node.terminal:
            feature_val = instance[node.feature]
            # Protection check
            if not node.children_values:
                break
            if feature_val not in node.children_values:
                break
            # Find the child node corresponding to the instance's feature value
            child_index = node.children_values.index(feature_val)
            if child_index == -1:
                break
            node = node.children[child_index]
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        # Get the size of the dataset
        dataset_size = len(dataset)
        # Initialize counter for correct predictions
        correct_prediction = 0
        # Iterate through each instance in the dataset
        for instance in dataset:
            # Check if the prediction matches the actual label
            if self.predict(instance) == instance[-1]:
                correct_prediction += 1
        # calculate the accuracy
        accuracy = (correct_prediction/dataset_size) * 100
        return accuracy
        
    def depth(self):
        return self.root.depth()
    
    def tree_depth(self, node):
        """
        Calculate the depth of the tree starting from the given node.

        Parameters:
        - node: The starting node of the tree.

        Returns:
        - depth: The depth of the tree.
        """
        # Base cases:
        if not node or node.terminal or not node.children:
            return 0
        # Recursive case:
        # Calculate the depth of each child subtree recursively
        # Take the maximum depth among all child subtrees and add 1 for the current node
        return 1 + max(self.tree_depth(child) for child in node.children)

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = DecisionTree(data=X_train, impurity_func=calc_entropy, max_depth=max_depth, gain_ratio=True) # entropy and gain ratio
        tree.build_tree()
        training.append(tree.calc_accuracy(X_train))
        validation.append(tree.calc_accuracy(X_validation))
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []

    for chi_val in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        tree = DecisionTree(data=X_train, impurity_func=calc_entropy, chi=chi_val, gain_ratio=True) # entropy and gain ratio
        tree.build_tree()
        # Calculate training and testing accuracies
        chi_training_acc.append(tree.calc_accuracy(X_train))
        chi_testing_acc.append(tree.calc_accuracy(X_test))
        # Calculate tree depth
        depth.append(tree.tree_depth(tree.root)) 
    return chi_training_acc, chi_testing_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    queue = [node]
    n_nodes = 0
    # Perform BFS traversal to visit each node in the tree
    while len(queue) > 0:
        # Dequeue a node from the queue
        current_node = queue.pop(0)
        # Increment the node count for the dequeued node
        n_nodes+=1
        # Enqueue the children of the current node
        queue.extend(current_node.children or [])
    return n_nodes

def calc_chi_square(groups, unique_values, counts):
    """
    Calculate the chi-square value for pruning.

    Parameters:
    - groups: Dictionary containing subsets of data after splitting.
    - unique_values: Unique class labels.
    - counts: Count of each class label in the dataset.

    Returns:
    - chi_square: The calculated chi-square value.
    """
    instances_count = np.sum(counts)
    chi_square = 0.0
    for feature, data_subset in groups.items():
        subset_size = data_subset.shape[0]

        for index, value in enumerate(unique_values):
            observed = np.sum(data_subset[:,-1] == value)
            expected = subset_size * (counts[index] / instances_count)
            chi_square += (((observed - expected) ** 2)/expected)
    return chi_square






