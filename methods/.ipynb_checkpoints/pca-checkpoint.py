import numpy as np

class PCA(object):
    """
        PCA dimensionality reduction object.
        Feel free to add more functions to this class if you need.
        But make sure that __init__, find_principal_components, and reduce_dimension work correctly.
    """
    def __init__(self, *args, **kwargs):
        """
            You don't need to initialize the task kind for PCA.
            Call set_arguments function of this class.
        """
        self.set_arguments(*args, **kwargs)
        #the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        #the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            The PCA class should have a variable defining the number of dimensions (d).
            You can either pass this as an arg or a kwarg.
        """
        self.d = kwargs["d"] if "d" in kwargs else (args[0] if len(args) > 0 else 1)


    def find_principal_components(self, training_data):
        """
            Finds the principal components of the training data. Returns the explained variance in percentage.
            IMPORTANT: 
            This function should save the mean of the training data and the principal components as
            self.mean and self.W, respectively.

            Arguments:
                training_data (np.array): training data of shape (N,D)
            Returns:
                exvar (float): explained variance
        """

        # Computes the mean of the training data
        self.mean = np.mean(training_data, axis=0)
        
        # Center the data by subtracting the mean
        training_data_centered = training_data - self.mean
        
        # Computes the Covariance Matrix
        covariance_matrix = np.cov(training_data_centered, rowvar = False)
        # Project the data onto the lower-dimensional space
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sorts the eigenvalues and the eigenvectors in descending order
        sorted_index = np.argsort(eigenvalues)[::-1]
 
        eigenvalues = eigenvalues[sorted_index]
        eigenvectors = eigenvectors[:, sorted_index]
        
        
        # Selects the top d eigenvectors
        self.W = eigenvectors[:, :self.d]
        
        # Computes the Explained Covariance
        exvar = 100 * np.sum(eigenvalues[:self.d]) / np.sum(eigenvalues)
        
        return exvar

    def reduce_dimension(self, data):
        """
            Reduce the dimensions of the data, using the previously computed
            self.mean and self.W. 

            Arguments:
                data (np.array): data of shape (N,D)
            Returns:
                data_reduced (float): reduced data of shape (N,d)
        """
        
        # Center the data by subtracting the mean
        data_centered = data - self.mean
        
        # Project the data onto the lower-dimensional space
        data_reduced = np.dot(data_centered, self.W)

        return data_reduced
        

