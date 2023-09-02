import numpy as np
from metrics import accuracy_fn, mse_fn, macrof1_fn

def splitting_fn(data, labels, indices, fold_size, fold):
    """
        Function to split the data into training and validation folds.
        Arguments:
            data (np.array, of shape (N, D)): data (which will be split to training 
                and validation data during cross validation),
            labels  (np.array, of shape (N,)): the labels of the data
            indices: (np.array, of shape (N,)): array of pre shuffled indices (integers ranging from 0 to N)
            fold_size (int): the size of each fold
            fold (int): the index of the current fold.
        Returns:
            train_data, train_label, val_data, val_label (np. arrays): split training and validation sets
    """
    
    split_indexes = np.array_split(indices,len(indices)//fold_size)
    fold_indexes = split_indexes[fold]

    mask_fold= np.zeros(data.shape[0], dtype=bool)
    mask_fold[fold_indexes] = 1

    mask_rest = ~mask_fold

    train_data = data[mask_rest]
    train_label = labels[mask_rest]
    val_data = data[mask_fold]
    val_label = labels[mask_fold]

    return train_data, train_label, val_data, val_label

def cross_validation(method_obj=None, search_arg_name=None, search_arg_vals=[], data=None, labels=None, k_fold=4):
    """
        Function to run cross validation on a specified method, across specified arguments.
        Arguments:
            method_obj (object): A classifier or regressor object, such as KNN. Needs to have
                the functions: set_arguments, fit, predict.
            search_arg_name (str): the argument we are trying to find the optimal value for
                for example, for DummyClassifier, this is "dummy_arg".
            search_arg_vals (list): the different argument values to try, in a list.
                example: for the "DummyClassifier", the search_arg_name is "dummy_arg"
                and the values we try could be [1,2,3]
            data (np.array, of shape (N, D)): data (which will be split to training 
                and validation data during cross validation),
            labels  (np.array, of shape (N,)): the labels of the data
            k_fold (int): number of folds
        Returns:
            best_hyperparam (float): best hyper-parameter value, as found by cross-validation
            best_acc (float): best metric, reached using best_hyperparam
    """
    ## choose the metric and operation to find best params based on the metric depending upon the
    ## kind of task.
    metric = mse_fn if method_obj.task_kind == 'regression' else macrof1_fn
    find_param_ops = np.argmin if method_obj.task_kind == 'regression' else np.argmax

    N = data.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)
    fold_size = N//k_fold

    acc_list1 = []
    for arg in search_arg_vals:
        arg_dict = {search_arg_name: arg}
        # this is just a way of giving an argument 
        # (example: for DummyClassifier, this is "dummy_arg":1)
        method_obj.set_arguments(**arg_dict)

        acc_list2 = []
        for fold in range(k_fold):
            #For to calculate the average error for a given Arg
            newValues = splitting_fn(data,labels,indices, fold_size, fold)
            train_data = newValues[0]
            train_label = newValues[1]
            val_data = newValues[2]
            val_label = newValues[3]
            method_obj.fit(train_data,train_label)
            #Fills the acc_list_2 in order to then find the average error
            acc_list2.append(metric(method_obj.predict(val_data), val_label))
        #Finds the average error and puts it into the acc_list_1
        averageError = np.sum(acc_list2)/k_fold
        acc_list1.append(averageError)
    #Fins the index of the best arg based on its average error with the cross-validation.
    bestHyperArgIndex = find_param_ops(acc_list1)
    best_hyperparam = search_arg_vals[bestHyperArgIndex]
    best_acc = acc_list1[bestHyperArgIndex]
    return best_hyperparam, best_acc

        


    