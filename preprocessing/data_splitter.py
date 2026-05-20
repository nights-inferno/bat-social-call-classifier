
import numpy as np
from dataset import AugmentationPipeline
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def balanced_test_data_train(X,y, test_size=0.2, random_state=42):
    #First, split the data using an iterative stratifier
    #Split the data into balanced training and test set
    kf = MultilabelStratifiedKFold(n_splits=1/test_size, shuffle=True, random_state=random_state) 
    train_idx, test_idx = next(kf.split(X, y))

    #Check the smallest class in test set
    y_test = y[test_idx]
    classes = y.shape[1]
    counts = np.sum(y_test, axis=0)
    min_class = np.argmin(counts)
    min_count = counts[min_class]

    #randomly remove samples from test set and add to training set until all classes have maximum of min_count samples
    while np.any(counts > min_count+1):
        for i in range(classes):
            if counts[i] > min_count+1:
                #Find indices of samples in test set that belong to class i
                class_indices = np.where(y_test[:, i] == 1)[0]
                #Randomly select one index to remove
                remove_idx = np.random.choice(class_indices)
                #Remove the sample from test set and add to training set
                train_idx = np.append(train_idx, test_idx[remove_idx])
                test_idx = np.delete(test_idx, remove_idx)
                y_test = y[test_idx]
                counts = np.sum(y_test, axis=0)
    
    return train_idx, test_idx