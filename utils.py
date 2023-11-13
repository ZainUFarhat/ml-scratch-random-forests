# numpy
import numpy as np

# Calculate accuracy - out of 100 examples, what percentage does our model get right?
def accuracy_fn(y_true, y_pred):
  """
  calculates the accuracies of a given prediction

  Parameters:

    y_true: the true labels
    y_pred: our predicted labels
  
  Returns:

    accuracy
  """
  
  # find the number of correct predictions  
  correct = np.equal(y_true, y_pred).sum()
  # calculate the accuracy
  acc = (correct/len(y_pred))*100
  # return the accuracy
  return round(acc, 2) 