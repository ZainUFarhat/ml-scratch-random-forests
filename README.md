# ml-scratch-random-forests
Random Forest Algorithm 

## **Description**
The following is my from scratch implementation of the Random Forest algorithm.

### **Dataset**

I tested the performance of my model on three datasets: \
\
    &emsp;1. Breast Cancer Dataset \
    &emsp;2. Iris Dataset \
    &emsp;3. Diabetes Dataset

### **Walkthrough**

**1.** Need the following packages installed: sklearn, numpy, collections, graphviz, and matplotlib.

**2.** Once you made sure all these libraries are installed, evrything is simple, just head to main.py and execute it.

**3.** Since code is modular, main.py can easily: \
\
    &emsp;**i.** Load the three datasets \
    &emsp;**ii.** Split data into train and test sets \
    &emsp;**iii.** Build a random forest classifier \
    &emsp;**iv.** Fit the random forest classifier \
    &emsp;**v.** Predict on the test set \
    &emsp;**vi.** Plot individual decision trees and corresponding forest.

**4.** In main.py I specify a set of hyperparameters, these can be picked by the user. The main ones worth noting are the number of trees, minimum samples split, and maximum depth. These hyperparameters were chosen through trail & error experimentation on each dataset.

### **Results**

For each dataset I will list the number of trees, minimum samples split values, maximum depth, and test Accuracy score.
In addition I offer some individual decision trees and random forests for visualization.

**1.** Breast Cancer Dataset:

- Hyperparameters:
     - Number of Trees = 5
     - Minimum Samples Split = 2
     - Maximum Depth = 5
 
- Numerical Result:
     - Accuracy = 96.49%

- See visualizations below:

    I will show each individual decision tree for this dataset only.


![alt text](https://github.com/ZainUFarhat/ml-scratch-random-forests/blob/main/plots/bc/decision_trees/bc_tree0.png?raw=true)
![alt text](https://github.com/ZainUFarhat/ml-scratch-random-forests/blob/main/plots/bc/decision_trees/bc_tree1.png?raw=true)
![alt text](https://github.com/ZainUFarhat/ml-scratch-random-forests/blob/main/plots/bc/decision_trees/bc_tree2.png?raw=true)
![alt text](https://github.com/ZainUFarhat/ml-scratch-random-forests/blob/main/plots/bc/decision_trees/bc_tree3.png?raw=true)
![alt text](https://github.com/ZainUFarhat/ml-scratch-random-forests/blob/main/plots/bc/decision_trees/bc_tree4.png?raw=true)
![alt text](https://github.com/ZainUFarhat/ml-scratch-random-forests/blob/main/plots/bc/bc_forest.png?raw=true)

**2.** Iris Dataset:

- Hyperparameters:
     - Number of Trees = 5
     - Minimum Samples Split = 2
     - Maximum Depth = 5
 
- Numerical Result:
     - Accuracy = 100.0%

- See visualization below:

![alt text](https://github.com/ZainUFarhat/ml-scratch-random-forests/blob/main/plots/iris/iris_forest.png?raw=true)

**2.** Diabetes Dataset:

- Hyperparameters:
     - Number of Trees = 10
     - Minimum Samples Split = 5
     - Maximum Depth = 3
 
- Numerical Result:
     - Accuracy = 76.4%

- See visualization below:

![alt text](https://github.com/ZainUFarhat/ml-scratch-random-forests/blob/main/plots/db/db_forest.png?raw=true)