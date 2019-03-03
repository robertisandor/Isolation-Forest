# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
import numpy as np
import pandas as pd 
from sklearn.metrics import confusion_matrix

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees 
        self.n_nodes = 0
        self.trees = []

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        height_limit = np.ceil(np.log2(self.sample_size))
        sample_indices = [np.random.choice(len(X), self.sample_size, replace=False) for idx in range(self.n_trees)]
        for idx in range(self.n_trees):
            tree = IsolationTree(height_limit=height_limit)
            tree.root = tree.fit(X[sample_indices[idx]], improved)
            self.trees.append(tree)
            self.n_nodes += tree.n_nodes 

        return self

    def path_length(self, x_i:np.ndarray, tree, e=0) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        if isinstance(tree, ExNode):
            if tree.n_rows > 2:
                c_t = (2.0 * (np.log(tree.n_rows - 1) + np.euler_gamma)) - (2.0 * (tree.n_rows - 1) / tree.n_rows)
            elif tree.n_rows == 2:
                c_t = 1
            else:
                c_t = 0
            return e + c_t 

        a = tree.split_attr 
        
        if x_i[a] < tree.split_value:
            return self.path_length(x_i, tree.left, e+1)
        else:
            return self.path_length(x_i, tree.right, e+1)

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        avg_lengths = [np.mean([self.path_length(X[idx, :], tree.root) for tree in self.trees]) for idx in range(X.shape[0])]
        avg_lengths = np.array(avg_lengths).reshape(X.shape[0], 1)

        c_n = (2.0 * (np.log(self.sample_size - 1) + np.euler_gamma)) - (2.0 * ((self.sample_size - 1) / self.sample_size))
        return np.array(2.0 ** (-1 * avg_lengths / c_n))        

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return (scores >= threshold)

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        return self.predict_from_anomaly_scores(self.anomaly_score(X), threshold)

class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.n_nodes = 0
        self.root = None

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.
        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        self.root = self.fit_(X, e=0)
        self.n_nodes += 1
        return self.root 

    def fit_(self, X, e):
        if e >= self.height_limit or X.shape[0] <= 1:
            node = ExNode(left=None, right=None, n_rows=X.shape[0])
            self.n_nodes += 1
        else: 
            # q is the split attribute
            # I choose a random column then calculate the min and max of that column
            q = np.random.randint(0, X.shape[1])
            col_min, col_max = np.min(X[:, q]), np.max(X[:, q])

            # p is the split value;
            # I choose a random value between the min and max
            # of the split_attr column 
            # and split the data into left and right halves accordingly
            p = np.random.uniform(col_min, col_max)
            X_l = X[X[:, q] < p]
            X_r = X[X[:, q] >= p]

            # if the column min and max are the same, 
            # then we have a dataframe of all of the same values 
            if col_min == col_max:
                node = ExNode(left=None, right=None, n_rows=X.shape[0])
                self.n_nodes += 1
                return node 

            node = InNode(split_attr=q, split_value=p, n_rows=X.shape[0])
            self.n_nodes += 1
            node.left = self.fit_(X_l, e+1)
            node.right = self.fit_(X_r, e+1)
        return node 


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    ...
    score_threshold = 1.0
    TPR = 0
    FPR = 0
    increment = 0.01
    while TPR < desired_TPR and (score_threshold - increment) > 0:
        score_threshold -= increment
        y_pred = (scores >= score_threshold)
        confusion = confusion_matrix(y, y_pred)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
    return score_threshold, FPR 
    

class ExNode:
    def __init__(self, left=None, right=None, n_rows=0):
        self.left = left 
        self.right = right 
        self.n_rows = 0
        
class InNode:
    def __init__(self, left=None, right=None, split_attr=None, split_value=None, n_rows=0):
        self.left = left
        self.right = right 
        self.n_rows = n_rows
        self.split_attr = split_attr
        self.split_value = split_value