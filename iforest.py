
# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
import numpy as np
import pandas as pd 
from random import randint 
from sklearn.metrics import confusion_matrix

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size 
        self.n_trees = n_trees 
        self.trees = []
        self.path_lengths = None

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            X = np.ndarray(buffer=X, shape=X.shape)
        self.path_lengths = np.zeros(shape=(X.shape[0], self.n_trees))
        height_limit = np.ceil(np.log2(self.sample_size))
        self.trees = [IsolationTree(height_limit=height_limit) for tree in range(self.n_trees)]
        self.samples = [np.random.choice(list(range(len(X))), self.sample_size, replace=False) for tree in range(self.n_trees)]
        for idx, tree in enumerate(self.trees):
            tree.fit(X[self.samples[idx]], improved)

        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        for idx, x_i in enumerate(X):
            for tree_idx, tree in enumerate(self.trees):        
                curr = tree.root
                a = tree.root.split_attr
                e = 0
                while e < tree.height_limit:
                    if x_i[a] < curr.split_value:
                        curr = curr.left
                    else:
                        curr = curr.right
                    e += 1
                    if curr.left is not None and curr.right is not None:
                        a = curr.split_attr
                    else:
                        break 
                    
                #print(e, curr.n_rows)
                estimated_path_length = 0
                if curr.n_rows > 2:
                    estimated_path_length = (2 * (np.log2(curr.n_rows - 1) + np.euler_gamma)) - (2 * (curr.n_rows - 1) / curr.n_rows)
                elif curr.n_rows == 2:
                    estimated_path_length = 1
                else:
                    estimated_path_length = 0
                # print(f"e = {e}, curr.n_rows = {curr.n_rows}, estimated_path_length = {estimated_path_length}, e + estimated_path_length = {e + estimated_path_length}")
                self.path_lengths[idx, tree_idx] = e + estimated_path_length
        #print(np.unique(self.path_lengths))
        return np.array(np.mean(self.path_lengths, axis=1)).reshape(len(X), 1)

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        X = np.ndarray(buffer=np.array(X), shape=X.shape)
        c_n = (2. * np.log2(self.sample_size - 1) + np.euler_gamma) - (2. * ((self.sample_size - 1) / self.sample_size))
        if np.sum(self.path_lengths) == 0:
            scores = np.array(2.**(-1. * (self.path_length(X) / c_n)), dtype=np.float16)
        else:
            scores = self.path_lengths
        return scores

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """ 
        # print(scores)
        return (scores >= threshold)

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        """A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."""
        return self.predict_from_anomaly_scores(self.anomaly_score(X), threshold)

# compute path lengths at the time of creation 
class IsolationTree:
    def __init__(self, height_limit, e=0):
        self.root = None
        self.height_limit = height_limit
        self.n_nodes = 0 
        self.e = e

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        if improved:
            return self.root
        else:
            if self.e >= self.height_limit or X.shape[0] <= 1:
                self.root = IsolationTreeNode(left=None, right=None, n_rows=X.shape[0])
                self.n_nodes += self.root.n_nodes
            else:
                q = randint(0, X.shape[1]-1)
                col_min, col_max = np.min(X[:, q]), np.max(X[:, q])
                p = np.random.uniform(col_min, col_max)
                X_l = X[X[:, q] < p]
                X_r = X[X[:, q] >= p]
                    
                while len(X_l) == 0 or len(X_r) == 0:
                    q = randint(0, X.shape[1]-1)
                    col_min, col_max = np.min(X[:, q]), np.max(X[:, q])
                    p = np.random.uniform(col_min, col_max)
                    X_l = X[X[:, q] < p]
                    X_r = X[X[:, q] >= p]

                # if len(np.unique(X_l[:, q])) == 1 or len(np.unique(X_r[:, q])) == 1:
                #     self.root = IsolationTreeNode(left=IsolationTreeNode(left=None, right=None, split_attr=q, split_value=p, n_rows=len(X_l)),
                #                               right=IsolationTreeNode(left=None, right=None, split_attr=q, split_value=p, n_rows=len(X_r)),
                #                               split_attr=q,
                #                               split_value=p,
                #                               n_rows=len(X))
                #     self.n_nodes += self.root.n_nodes
                #     return self.root 

                self.root = IsolationTreeNode(left=IsolationTree(height_limit=self.height_limit, e=self.e+1).fit(X_l, improved),
                                              right=IsolationTree(height_limit=self.height_limit, e=self.e+1).fit(X_r, improved),
                                              split_attr=q,
                                              split_value=p,
                                              n_rows=len(X))
                self.root.left.n_rows = len(X_l)
                self.root.right.n_rows = len(X_r)
                self.n_nodes += self.root.left.n_nodes + self.root.right.n_nodes   
        return self.root


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
        print(f"TN, FP, FN, TP = {TN}, {FP}, {FN}, {TP}")
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        print(f"score_threshold = {score_threshold}, TPR = {TPR}, FPR = {FPR}")
    return score_threshold, FPR 

class IsolationTreeNode:
    def __init__(self, left, right, split_attr=None, split_value=None, n_nodes=1, n_rows=0):
        self.left = left 
        self.right = right 
        self.split_attr = split_attr 
        self.split_value = split_value 
        self.n_nodes = n_nodes 
        if left is not None:
            self.n_nodes += self.left.n_nodes
        if right is not None:
            self.n_nodes += self.right.n_nodes
        self.n_rows = n_rows