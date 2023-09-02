import numpy as np

class KNN(object):
    """
        kNN classifier object.
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
            The KNN class should have a variable defining the number of neighbours (k).
            You can either pass this as an arg or a kwarg.
        """
        self.k = kwargs["k"] if "k" in kwargs else (args[0] if len(args) > 0 else 1)
    
    def euclidean_dist(self, example):
        """function to compute the Euclidean distance between a single example
        vector and all training_examples

        Inputs:
            example: shape (D,)
            training_examples: shape (NxD) 
        Outputs:
            return distance vector of length N
        """
        
        # Computes the euclidean distance between a single example vector and all of the training examples vectors
        distances = np.sqrt(np.sum(np.square(example - self.training_data), axis = 1))
        
        return distances
    
    def find_k_nearest_neighbors(self, distances):
        """ Find the indices of the k smallest distances from a list of distances.
            Tip: use np.argsort()
        """
        # Sorts the distances by increasing values
       
        indices = np.argpartition(distances, self.k)[:self.k]
        # Selects the k smallest distances
        
        return indices
    
    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.
            
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """
        # Stores the training data as part of the class
        self.training_data = training_data
        
        # Stores the training labels as part of the class
        self.training_labels = training_labels 
        
        return self.training_labels
    
    def predict_single(self, test_data_single):
        
        # Computes the euclidean distance between the test data and the training data
        distances = self.euclidean_dist(test_data_single) 
        
        # Sorts the distances by increasing distance.
        nn_indices = self.find_k_nearest_neighbors(distances)
        
        # Selects the correct labels for the given indices
        neighbor_labels = self.training_labels[nn_indices] 
        
        # Computes the mean of these labels
        mean_label = neighbor_labels.mean(axis=0)
        return mean_label

    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        
        # Computes the prediction label for each axis of the test data matrix of shape (N, D)
        test_labels = np.apply_along_axis(func1d=self.predict_single, axis=1, arr=test_data)
        
        return test_labels