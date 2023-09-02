import numpy as np
import sys
sys.path.append('..')
from utils import label_to_onehot


def f_softmax(data, w):
    return (np.exp(np.dot(data, w.T).T) / np.sum(np.exp(np.dot(data, w.T).T))).T

def sigmoid(t):
    """ Sigmoid function
    
    Args:
        t (np.array): Input data of shape (N, )
        
    Returns:
        res (np.array): Probabilites of shape (N, ), where each value is in [0, 1].
    """
    
    
    return 1/(1 + np.exp(-t))
    
def gradient_logistic_multi(data, labels, w):
    return data.T @ (f_softmax(data, w) - labels)

def gradient_logistic(data, labels, w):
    """ Logistic regression gradient function for binary classes
    
    Args:
        data (np.array): Dataset of shape (N, D).
        labels (np.array): Labels of shape (N, ).
        w (np.array): Weights of logistic regression model of shape (D, )
    Returns:
        grad (np. array): Gradient array of shape (D, )
    """
    return data.T.dot(sigmoid(data.dot(w)) - labels)


def logistic_regression_classify(data, w):
    """ Classification function for binary class logistic regression. 
    
    Args:
        data (np.array): Dataset of shape (N, D).
        w (np.array): Weights of logistic regression model of shape (D, )
    Returns:
        predictions (np.array): Label assignments of data of shape (N, )
    """
    predictions = sigmoid(data @ w)
    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1
    return predictions

def accuracy_fn(labels_gt, labels_pred):
    """ Computes accuracy.
    
    Args:
        labels_gt (np.array): GT labels of shape (N, ).
        labels_pred (np.array): Predicted labels of shape (N, ).
        
    Returns:
        acc (float): Accuracy, in range [0, 1].
    """
    
    return np.sum(labels_pred == labels_gt) / labels_gt.shape[0]

def logistic_regression_train_multi(data, labels, max_iters, lr):
        if len(labels.shape) > 1:
            N = labels.shape[1]
        else:
            N = 1
        weights = np.random.normal(0, 0.1, [data.shape[1], N])
        for it in range(max_iters):
            gradient = gradient_logistic_multi(data, labels, weights)
            weights = weights - lr * gradient
            predictions = logistic_regression_classify_multi(data, weights)
            if accuracy_fn(helpers.onehot_to_label(labels), predictions) == 1:
                break
        return weights
    
    
def logistic_regression_train(data, labels, max_iters, lr):
    """ Training function for binary class logistic regression. 
    
    Args:
        data (np.array): Dataset of shape (N, D).
        labels (np.array): Labels of shape (N, ).
        max_iters (integer): Maximum number of iterations. Default:10
        lr (integer): The learning rate of  the gradient step. Default:0.001
        print_period (int): Num. iterations to print current loss. 
            If 0, never printed.
        plot_period (int): Num. iterations to plot current predictions.
            If 0, never plotted.
    Returns:
        np.array: weights of shape(D, )
    """

    #initialize the weights randomly according to a Gaussian distribution
    weights = np.random.normal(0., 0.1, [data.shape[1],])
    for it in range(max_iters):
        gradient = gradient_logistic(data, labels, weights)
        weights = weights - lr*gradient
        predictions = logistic_regression_classify(data, weights)
        if accuracy_fn(labels, predictions) == 1:
            break
    return weights

class LogisticRegression(object):
    """
        LogisticRegression classifier object.
        Feel free to add more functions to this class if you need.
        But make sure that __init__, set_arguments, fit and predict work correctly.
    """

    def __init__(self, *args, **kwargs):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        
        self.task_kind = "classification"
        self.set_arguments(*args, **kwargs)

    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            The LogisticRegression class should have variables defining the learning rate (lr)
            and the number of max iterations (max_iters)
            You can either pass these as args or kwargs.
        """
        
        self.lr = kwargs["lr"] if "lr" in kwargs else (args[0] if len(args) > 0 else 1)
        #self.max_iters = kwargs["max_iters"] if "max_iters" in kwargs else (args[0] if len(args) > 0 and "lr" in kwargs else (args[1] if len(args) > 1 else 1))
        self.max_iters = 1000
    
    def predict(self, test_data):  
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
       regression     Returns:
                test_labels (np.array): labels of shape (N,)
        """   

        if len(self.weights.shape) > 1:
            predictions = f_softmax(test_data, self.weights)
        else:
            predictions = logistic_regression_classify(test_data, self.weights)
        return predictions


    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        
        if len(training_labels.shape) > 1:
            self.weights = logistic_regression_train_multi(training_data, training_labels, self.max_iters, self.lr)
        else:
            self.weights = logistic_regression_train(training_data, training_labels, self.max_iters, self.lr)
        return self.predict(training_data)

    