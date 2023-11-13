# os
import os

# Decision Trees
from DecisionTrees import *

# graphviz
from graphviz import Source

# matplotlib
import matplotlib.pyplot as plt

# random forest
class RandomForests():

    """
    Description:
        My form scratch implementation of the Decision Trees Algorithm
    """

    # constructor
    def __init__(self, num_trees, max_depth, min_samples_split, num_features = None):

        """
        Description:
            Constructor of our Random Forests class
        
        Parameters:
            num_trees: number of decision trees to have in our forest
            max_depth: maximum dept allowed for ecah tree
            min_samples_split: stepping criteria, minimum number of samples to split a node on (specific for a decision tree)
            num_features: adds some randomness to decision tree by selecting subset of features
        
        Returns:
            None
        """

        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_features = num_features

    # fit
    def fit(self, X, y):

        """
        Description:
            Fits our Random Forests class to the number of decision trees given during class instantiation
        
        Paramters:
            X: train features
            y: train labels
        
        Return:
            None
        """

        # create an empty list to store all our decision trees
        self.decision_trees = []

        # iterate over the number of trees we chose to create
        for _ in range(self.num_trees):

            # create an instance for our current decision tree
            decision_tree = DecisionTree(max_depth = self.max_depth, min_samples_split = self.min_samples_split, 
                                                                                                num_features = self.num_features)
            
            # select a random sample from our training dataset
            # how this works is each decision tree will be trained on a different sample of the training dataset
            # however, what is different here is some training samples can occur multiple times in the sample
            # this is because we set replace to True
            # this is the "randomness" of the random forests

            X_sample, y_sample = self.bootstrap_samples(X, y)

            # fit our current decision tree on the selected subset
            decision_tree.fit(X_sample, y_sample)

            # append our decision tree to the decision trees list
            self.decision_trees.append(decision_tree)
    
    # bootstrap samples
    def bootstrap_samples(self, X, y):

        """
        Description:
            Selects a random sample of the training set
        
        Parameters:
            X: train set features
            y: train set labels
        
        Returns:
            X_sample, y_sample
        """

        # fetch the number of samples
        num_samples = X.shape[0]

        # generate a set of random indices
        # note: the same index can occur multiple times in our generated indices
        # this is the randomness we are adding, else all decision trees will be identical
        indices = np.random.choice(num_samples, num_samples, replace = True)

        # generate our samples
        X_sample, y_sample = X[indices], y[indices]

        # return
        return X_sample, y_sample

    # most common label
    def most_common_label(self, y):

        """
        Description:
            Decides the label of our leaf node
        
        Parameters:
            y: labels
        
        Returns:
            predicted_label
        """

        # create a counter for our given labels
        counter = Counter(y)
        # we just need the most common label from our counter
        predicted_label = counter.most_common(1)[0][0]

        # return
        return predicted_label
    
    # predict
    def predict(self, X):

        """
        Description:
            Predict on the fitted random forest
        
        Parameters:
            X: test set
        
        Returns:
            predictions
        """

        # generate an array that holds our predictions for all decision trees in our random forest
        predictions = np.array([decision_tree.predict(X) for decision_tree in self.decision_trees])

        # we will have an array that holds the predictions of all samples for each tree
        # i.e
        # predictions = [[0, 1 , 1, 0, 1,...], [1, 1 , 0, 0, 1,...], [1, 0 , 1, 0, 1,...], ...]
        # the index of each subarray corresponds to the prediction of a certain row of features for each decision tree
        # in order to achive a majority vote for each sample, so we can have one prediction for every row of features
        # we will swap the axes in our predictions array, which means every subarray now holds the votes for each row of features
        # if our predictions was of shape m x n where m is # trees and n # samples
        # not it will be n x m where each row corresponds to that specific sample
        # each row now has m elements corresponding to each trees predicition for that specific sample
        # it is not easy to get the majority vote, we just need to find the most common label among all labels in subarray

        tree_pedictions = np.swapaxes(predictions, 0, 1)

        # update our predictions
        predictions = np.array([self.most_common_label(pred) for pred in tree_pedictions])

        # return
        return predictions
    
    def visualize_forest(self, feature_names, class_names, title, tree_savepath, forest_savepath):

        """
        Description:
            Plot all the decision trees of all our forests
        
        Parameters:
            feature_names: names of features to add to nodes in our trees
            class_names: names of labels for leaf nodes
            title: title of our random forest plot
            tree_savepath: path to save our decision trees to
            forest_savepath: path to save our random forest to
        
        Returns:
            None
        """

        # iterate through all the decision trees of our random forest
        for (i, tree) in enumerate(self.decision_trees):

            # plot the decision trees
            tree_dot = tree.tree_to_dot(feature_names, class_names, title=f'Tree {i}')
            graph = Source(tree_dot, format = 'png')
            graph.render(filename = tree_savepath + f'bc_tree{i}', directory = '.', cleanup = True, view = False)
        
        # we want to create our random forest grid
        # the best way to visualize our forest is by plotting all decision trees on a grid
        # we first calculate the number of rows and columns of our grid
        # this is based on the number of trees
        num_rows = int(self.num_trees**0.5)
        num_cols = (self.num_trees + num_rows - 1) // num_rows

        # create our figure and axes
        # choose a large enough size for our figure
        fig, axs = plt.subplots(num_rows, num_cols, figsize=  (15, 15))
        fig.suptitle(title, fontsize = 30, fontweight = 'bold')
        fig.set_facecolor("lavender")

        # iterate through our trees and add each of its plot to its corresponding grid
        for i in range(self.num_trees):
            img_path = os.path.join(tree_savepath, f'bc_tree{i}.png')
            img = plt.imread(img_path)
            row, col = divmod(i, num_cols)
            axs[row, col].imshow(img)
            axs[row, col].axis('off')
        
        # remove empty subplots
        for i in range(len(axs.reshape(-1))):
            if i >= self.num_trees:
                axs.flatten()[i].set_visible(False)

        # make the layout tight to avoid a lot of white space
        plt.tight_layout(rect = [0, 0, 1, 0.96])

        # save our forest
        plt.savefig(forest_savepath)     



