# numpy
import numpy as np

# collections
from collections import Counter

# nodes of decision tree 
class Node():

    """
    Description:
        Holds characteristics for each individual node in our Decision Tree
    """

    # constructor
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None):

        """
        Description:
            Constructor that assigns the charecteristics of each node in our Decision Tree
        
        Parameters:
            feature: the feature we split on
            threshold: the threshold to consider for splitting on feature above
            left: pointer to left subtree
            right: pointer to right subtree
            value: if it is a leaf node, what is the value? (or what is the class label?)

        Returns:
            None
        """

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    # check if our current node is a leaf node
    def check_leaf_node(self):
        """
        Description:
            Checks whether the current Node instance is a leaf node or not
        
        Parameters:
            None

        Returns:
            is_leaf
        """

        is_leaf = self.value is not None

        # return
        return is_leaf

# decision tree
class DecisionTree():

    """
    Description:
        My form scratch implementation of the Decision Trees Algorithm
    """

    # constructor
    def __init__(self, min_samples_split, max_depth, num_features = None):

        """
        Description:
            Constructor of our Decision Trees class
        
        Paramters:
            min_samples_split: stepping criteria, minimum number of samples to split a node on
            max_depth: maximum depth to allow our decision tree to grow to
            num_features: adds some randomness to decision tree by selecting subset of features
        
        Returns:
            None
        """

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.num_features = num_features

        # want to still have access to root of node
        # this is how we will start traversing the tree during inference time
        self.root = None
    
    # fit
    def fit(self, X, y):

        """
        Description:
            Fits our decision tree based on train features and labels
        
        Parameters:
            X: features of train set
            y: labels of train set
        
        Returns:
            None
        """

        # need to check that number of features does not exceed the number of features we already have
        # user might make the mistake of selecting num_features > X actual features
        # if in class initialization we did not define number of features we select all features of x
        # else it needs to be the minimum between X's number of features and the number of features the user defined
        self.num_features = X.shape[1] if not self.num_features else min(X.shape[1], self.num_features)
        
        # grow our tree
        self.root = self.grow_tree(X, y)
    
    # grow tree
    def grow_tree(self, X, y, depth = 0):

        """
        Description:
            Grows our tree from the root node specified in the fit function
        
        Parameters:
            X: features of train set
            y: labels of train set
            depth: the depth of our decision tree, intially zero and is increased by one for every child node created 
        
        Returns:
            None
        """

        # fetch number of samples and features from X
        N, n_features = X.shape
        # fetch number of labels from y
        num_labels = len(np.unique(y))

        # 1. Check the stopping criteria:

        # if we exceed maximum depth chosen
        # or we only have one label at our current node
        # or the number of samples is less than the minimum we are allowed to split on
        # then we are a leaf node
        if depth >= self.max_depth or num_labels == 1 or N < self.min_samples_split:

            # find the class label for leaf node
            leaf_value = self.most_common_label(y)

            # conclude this specific part of the tree
            return Node(value = leaf_value)

        # 2. Find the best split:

        # this is the part where we create the randomness in the decision trees
        # feature indices is just a random numpy array of the range of number of features
        feature_indices = np.random.choice(n_features, self.num_features, replace = False)
        # find the best threshold and feature based on random feature indices generated
        best_feature, best_threshold = self.best_split(X, y, feature_indices)

        # 3. Create child nodes:

        # get the left and right indices based on our best feature and threshold obtained from step 2
        left_indices, right_indices = self.split(X[:, best_feature], best_threshold)

        # recursively grow our tree to left and right
        left = self.grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right = self.grow_tree(X[right_indices, :], y[right_indices], depth + 1)

        # return 
        return Node(best_feature, best_threshold, left, right)
    
    # best splot
    def best_split(self, X, y, feature_indices):

        """
        Descriptions:
            Finds the best split based on features, labels, and the random feature indices created in grow_tree step #2
            Find threshold among all possible thresholds/splits that are out there, which one is the best?
        
        Parameters:
            X: features
            y: labels
            feature_indices: random feature indices generated in grow_tree method
        
        Returns:
            split_idx, split_threshold
        """

        # best gain value
        best_gain = -1
        # split indices and threshold
        split_idx, split_threshold = None, None

        # iterate through
        for feature_index in feature_indices:

            # get the feature column
            X_column = X[:, feature_index]
            # get all possible thresholds
            thresholds = np.unique(X_column)

            # now traverse all possible thresholds
            for thresh in thresholds:

                # calculate the information gain
                gain = self.information_gain(y, X_column, thresh)

                # check if this gain in information is better than what we already have
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_index
                    split_threshold = thresh
                
        # return
        return split_idx, split_threshold


    # information gain
    def information_gain(self, y, X_column, thresh):

        """
        Description:
            Calculates the information gain value for the specific split feature and threshold

        Parameters:
            y: labels
            X_column: current features we are considering for a possible splitting criterion
            thresh: current threshold we are considering for a possible splitting criterion 

        Returns:
            info_gain
        """

        # 1. Parent entropy:

        parent_entropy = self.entropy(y)

        # 2. Create children:

        left_indices, right_indices = self.split(X_column, thresh)

        # if the threshold was only met for either left or right
        # meaning one class did not meet the splitting criteria at all
        if len(left_indices) == 0 or len(right_indices) == 0:
            # then our information gain is 0
            return 0

        # 3. Calculate the weighted average entropy of children:

        # find the number of elements in left and right children created from split
        n = len(y)
        num_left, num_right = len(left_indices), len(right_indices)

        # find the entropy of left and right children
        entropy_left, entropy_right = self.entropy(y[left_indices]), self.entropy(y[right_indices])

        # weighted average of entropy means how many samples are in one child and how many are in the other
        child_entropy = (num_left / n) * entropy_left + (num_right / n) * entropy_right

        # 4. Calculate the information gain:

        info_gain = parent_entropy - child_entropy
    
        # return
        return info_gain
    
    # split
    def split(self, X_column, split_thresh):

        """
        Description:
            Create children based on feature and splitting threshold from best_split
        
        Parameters:
            X_column: current feature we are assessing
            split_thresh: current threshold we are assessing
        
        Returns:
            left_indices, right_indices
        """

        # need to find which indices will go to the left and right
        left_indices = np.argwhere(X_column <= split_thresh).flatten()
        right_indices = np.argwhere(X_column > split_thresh).flatten()

        # return
        return left_indices, right_indices



    # entropy
    def entropy(self, y):

        """
        Description:
            Calculate the entropy of a given node
        
        Parameters:
            y: labels at that particular node
        
        Returns:
            entropy_value
        """

        # calculate entropy value
        
        # create a histogram that counts the number of times each item in y occurs
        # this returns an array with a count of the number of times each label appeared
        hist = np.bincount(y)
        # calculate probability of each label occuring in given y
        p_x = hist / len(y)
        
        entropy_value = -1* np.sum([p * np.log(p) for p in p_x if p > 0 ])

        # return
        return entropy_value

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
            Predicts the labels for our unseen data
        
        Parameters:
            X: features of test set
        
        Returns:
            predicitons
        """

        # find predictions
        predicitons = np.array([self.traverse_tree(x, self.root) for x in X])

        # return 
        return predicitons

    # traverse tree
    def traverse_tree(self, x, node):

        """
        Description:
            Traverse the tree to find our label based on given feature
        
        Parameters:
            x: current row of X (this is all features of a single row in X)
            node: node to start traversing from
        
        Returns:
            label
        """

        # if we are at a leaf node
        if node.check_leaf_node():

            # return the node value, that is our class label
            label = node.value

            # return 
            return label
        
        # if the value for the selected feature of our splitting criterion is less than the threshold determined during training
        if x[node.feature] <= node.threshold:
            # traverse to left subtree
            return self.traverse_tree(x, node.left)

        # else, traverse to right subtree
        return self.traverse_tree(x, node.right)
    
    def tree_to_dot(self, feature_names, class_names, title):

        """
        Description:
            Converts our decision tree to dot format so we can visualize it with graphviz
        
        Parameters:
            feature_names: names of our features
            class_names: names of classes (labels)
        
        Returns:
            gviz_dot_string
        """

        # to increase our title font, we will work on converting point to pixel where 1 point = 1.33 pixels
        pixel_size = int(21 * 1.33)

        # what we will do is create a string called gviz_dot_string that will collect our dot data that we will use to render 
        # our graph with graphviz

        # create the string setting the labels, fonts, colors, title, etc.
        gviz_dot_string = f'digraph Tree {{\nlabel=<<font face="Arial" point-size="{pixel_size}"><b>{title}</b></font>>;\nlabelloc=t; \
                            \nrankdir=TB;\nmargin=0.1;\nnode [color="#000000", fontname="Arial", fontsize=10];\nedge [fontname="Arial"];\n'

        # I like to add lavender backgounds, just my style
        gviz_dot_string += 'graph [bgcolor="lavender"];\n'

        # just some dummy nodes to make sure my "root" does not have an arrow pointing above it
        # we will make the arrow invisible to resolve this
        gviz_dot_string += 'dummy [label="", width=0, style=invis];\n'
        gviz_dot_string += 'dummy -> root [label="", color="#000000", dir="none", style=invis];\n'

        # want the root node to have the same style and filling as the rest
        gviz_dot_string += 'root [fillcolor="orange", style=filled];\n'

        # create a queue
        # what we will do is perform a depth first search on our tree so we can achieve the best feature and its corresponding threshold
        queue = [(self.root, 'root')]

        # DFS
        while queue:

            # get the current node and its parent id
            node, parent_id = queue.pop(0)

            # check if this is a leaf node
            if node.check_leaf_node():

                # get the class label (our prediction)
                class_label = class_names[node.value]
                # check if our dataset only has two classes
                if len(class_names) == 2:
                    # color the label red for label 0, green for others
                    label_color = '#FF0000' if node.value == 0 else '#008000'
                else: # '#FFAAAA', '#AAFFAA', '#AAAAFF'
                    # color the label red for label 0, green for others
                    label_color = '#FFAAAA' if node.value == 0 else '#AAFFAA' if node.value == 1 else '#AAAAFF' 
                # append all this to our dot data string 
                gviz_dot_string += f'"{parent_id}" -> "{node.value}" [label="", color="#000000"];\n'
                gviz_dot_string += f'"{node.value}" [label="{class_label}", color="{label_color}", fillcolor="{label_color}", style=filled];\n'
            # if we are not on a leaf node (meaning, we have children)
            else:
                # get the name of the best feature used at current node
                feature_name = feature_names[node.feature]
                # also get the threshold
                threshold = node.threshold
                # set the text content inside of our node to our condition which is feature <= threshold
                current_id = f'{feature_name} <= {threshold:.2f}'
                # fill it with color orange
                fill_color = "#e58139" 
                # append all this to our dot data
                gviz_dot_string += f'"{parent_id}" -> "{current_id}" [label="", color="#000000"];\n'
                gviz_dot_string += f'"{current_id}" [label="{feature_name} <= {threshold:.2f}", color="#000000", fillcolor="{fill_color}", style=filled];\n'
                # append the left and right children to our queue
                # we will keep on repeating this until we reach all leaf nodes and our queue is empty
                queue.append((node.left, current_id))
                queue.append((node.right, current_id))

        # close our string with curly bracket
        gviz_dot_string += '}'

        # return
        return gviz_dot_string