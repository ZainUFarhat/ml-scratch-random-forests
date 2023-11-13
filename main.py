# datasets
from datasets import *

# Random Forest 
from RandomForests import *

# utils
from utils import *

# set numpy random seed
np.random.seed(42)

def main():

    """
    Description:
        Trains and tests our Random Forest
    
    Parameters:
        None
    
    Returns:
        None
    """

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    test_size = 0.2
    random_state = 42
    dataset_name = 'Breast Cancer'
    
    # create an instance of Datasets class
    datasets = Datasets(test_size = test_size, random_state = random_state)

    # load the breast cancer dataset
    feature_names, class_names, X_train, X_test, y_train, y_test = datasets.load_breast_cancer()

    print(f'Loading {dataset_name} Dataset...')
    print(f'\nThe Features of {dataset_name} Dataset are:', ', '.join(feature_names))
    print(f'The Labels of the {dataset_name} Dataset are:', ', '.join(class_names))
    print(f'\n{dataset_name} contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nRandom Forest Classifier\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    print('Training...\n')

    # decision tree hyperparameters
    num_trees = 5
    min_samples_split = 2
    max_depth = 5

    rf = RandomForests(num_trees = num_trees, min_samples_split = min_samples_split, max_depth = max_depth)
    rf.fit(X_train, y_train)

    print('Done Training!') 
    print('---------------------------------------------------Testing----------------------------------------------------')
    print('Testing...\n')
    predictions = rf.predict(X_test)

    acc = accuracy_fn(y_true = y_test, y_pred = predictions)

    print('{0} Test Accuracy = {1}%'.format(dataset_name, acc))
    print('\nDone Testing!')
    print('---------------------------------------------------Plotting---------------------------------------------------')
    print('Plotting...')
    title = f'{dataset_name} Random Forest'
    tree_savepath = 'plots/bc/decision_trees/'
    forest_savepath = 'plots/bc/bc_forest.png'

    # convert our tree to dot format so we can render it with graphviz
    rf.visualize_forest(feature_names = feature_names, class_names = class_names, title = title,
                        tree_savepath = tree_savepath, forest_savepath = forest_savepath)

    print('Please refer to plots/bc directory to view Random Forest.')
    print('--------------------------------------------------------------------------------------------------------------\n')

    ######################################################################################################################################

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    dataset_name = 'Iris'
    
    # create an instance of Datasets class
    datasets = Datasets(test_size = test_size, random_state = random_state)

    # load the iris dataset
    feature_names, class_names, X_train, X_test, y_train, y_test = datasets.load_iris()

    print(f'Loading {dataset_name} Dataset...')
    print(f'\nThe Features of {dataset_name} Dataset are:', ', '.join(feature_names))
    print(f'The Labels of the {dataset_name} Dataset are:', ', '.join(class_names))
    print(f'\n{dataset_name} contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nRandom Forest Classifier\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    print('Training...\n')

    # decision tree hyperparameters
    min_samples_split = 2
    max_depth = 5

    rf = RandomForests(num_trees = num_trees, min_samples_split = min_samples_split, max_depth = max_depth)
    rf.fit(X_train, y_train)

    print('Done Training!') 
    print('---------------------------------------------------Testing----------------------------------------------------')
    print('Testing...\n')
    predictions = rf.predict(X_test)

    acc = accuracy_fn(y_true = y_test, y_pred = predictions)

    print('{0} Test Accuracy = {1}%'.format(dataset_name, acc))
    print('\nDone Testing!')
    print('---------------------------------------------------Plotting---------------------------------------------------')
    
    print('Plotting...')
    title = f'{dataset_name} Random Forest'
    tree_savepath = 'plots/iris/decision_trees/'
    forest_savepath = 'plots/iris/iris_forest.png'

    # convert our tree to dot format so we can render it with graphviz
    rf.visualize_forest(feature_names = feature_names, class_names = class_names, title = title,
                        tree_savepath = tree_savepath, forest_savepath = forest_savepath)
    
    print('Please refer to plots/iris directory to view Random Forest.')
    print('--------------------------------------------------------------------------------------------------------------\n')
    #######################################################################################################################################

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    dataset_name = 'Diabetes'
    
    # create an instance of Datasets class
    datasets = Datasets(test_size = test_size, random_state = random_state)

    # load the diabetes dataset
    feature_names, class_names, X_train, X_test, y_train, y_test = datasets.load_diabetes()

    print(f'Loading {dataset_name} Dataset...')
    print(f'\nThe Features of {dataset_name} Dataset are:', ', '.join(feature_names))
    print(f'The Labels of the {dataset_name} Dataset are:', ', '.join(class_names))
    print(f'\n{dataset_name} contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nRandom Forest Classifier\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    print('Training...\n')

    # decision tree hyperparameters
    num_trees = 10
    min_samples_split = 5
    max_depth = 3

    rf = RandomForests(num_trees = num_trees, min_samples_split = min_samples_split, max_depth = max_depth)
    rf.fit(X_train, y_train)

    print('Done Training!') 
    print('---------------------------------------------------Testing----------------------------------------------------')
    print('Testing...\n')
    predictions = rf.predict(X_test)

    acc = accuracy_fn(y_true = y_test, y_pred = predictions)

    print('{0} Test Accuracy = {1}%'.format(dataset_name, acc))
    print('\nDone Testing!')
    print('---------------------------------------------------Plotting---------------------------------------------------')
    
    print('Plotting...')
    title = f'{dataset_name} Random Forest'
    tree_savepath = 'plots/db/decision_trees/'
    forest_savepath = 'plots/db/db_forest.png'

    # convert our tree to dot format so we can render it with graphviz
    rf.visualize_forest(feature_names = feature_names, class_names = class_names, title = title,
                        tree_savepath = tree_savepath, forest_savepath = forest_savepath)
    
    print('Please refer to plots/db directory to view Random Forest.')
    print('--------------------------------------------------------------------------------------------------------------')


    return None

if __name__ == '__main__':

    # run everything
    main()