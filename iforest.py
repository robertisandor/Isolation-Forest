
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

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        # if it's a dataframe
        if isinstance(X, pd.DataFrame):
            X = X.values
            X = np.ndarray(buffer=X, shape=X.shape)
        # TODO: figure out what to set height_limit argument to
        height_limit = int(np.floor(np.log2(self.sample_size)))
        self.trees = [IsolationTree(height_limit=height_limit) for tree in range(self.n_trees)]
        for tree in self.trees:
            tree.fit(X, improved)

        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        values = [np.mean([self.instance_path_length(X[idx, :], 
                                                    tree.root, 
                                                    height_limit=tree.height_limit, 
                                                    e=0, 
                                                    total_size=len(X)) 
                            for tree in self.trees]) for idx in range(len(X))]
        return values

    def instance_path_length(self, x_i, tree, height_limit, e=0, total_size=0):
        if e >= height_limit or (tree.left is None and tree.right is None):
            estimated_path_length = 0
            if tree.n_rows > 2:
                # c(size) = an adjustment for an unbuilt subtree (that went past the height_limit)
                # this part computes the estimate for exceeding the height limit
                estimated_path_length = 2 * (np.log(tree.n_rows - 1) + np.euler_gamma) - (2 * (tree.n_rows - 1) / tree.n_rows)
            elif tree.n_rows == 2:
                estimated_path_length = 1
            else:
                estimated_path_length = 0
            return e + estimated_path_length
        else:
            a = tree.split_attr
            if x_i[a] < tree.split_value:
                return self.instance_path_length(x_i, tree.left, height_limit, e+1)
            else:
                return self.instance_path_length(x_i, tree.right, height_limit, e+1)

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        # TODO: actually compute anomaly score 
        X = np.ndarray(buffer=np.array(X), shape=X.shape)
        # is it self.sample_size?
        c_n = 2 * (np.log(self.sample_size - 1) + np.euler_gamma) - 2 * (self.sample_size - 1) / self.sample_size
        avg_path_lengths = self.path_length(X)
        scores = np.array([2**(-1 * avg_path_lengths[idx] / c_n) for idx, x_i in enumerate(X)])
        print(pd.DataFrame(scores).describe())
        return np.ndarray(buffer=np.array(scores), shape=(len(scores), 1), dtype=np.float16)

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """ 
        return np.ndarray(buffer=np.array([1 if score < threshold else 0 for score in scores]), 
                            shape=(len(scores), 1), 
                            dtype=np.int16)

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        """A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."""
        return self.predict_from_anomaly_scores(self.anomaly_score(X), threshold)


class IsolationTree:
    def __init__(self, height_limit, e=0):
        self.root = None
        self.height_limit = height_limit
        self.n_nodes = 0 

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        #print("Fit!")
        # TODO: flesh this out
        if improved:
            return self.root
        else:
            if self.height_limit <= 0 or X.shape[0] <= 1:
                self.root = IsolationTreeNode(left=None, right=None, n_rows=X.shape[0])
                self.n_nodes += self.root.n_nodes
            else:
                q = randint(0, X.shape[1]-1)
                split_idx = randint(0, X.shape[0]-1)
                p = X[split_idx, q]
                col_min, col_max = np.min(X[:, q]), np.max(X[:, q])
                if col_min == col_max:
                    self.root = IsolationTreeNode(left=None, right=None)
                    self.n_nodes += self.root.n_nodes
                    self.root.n_rows = len(X)
                    return self.root
                # don't know if this portion below is valid/necessary
                # if p == col_min or p == col_max:
                #     self.root = IsolationTreeNode(left=None, right=None)
                #     self.n_nodes += self.root.n_nodes
                #     return self.root
                X_l = X[X[:, q] < p]
                X_r = X[X[:, q] >= p]
                self.root = IsolationTreeNode( # should these be isolation trees or isolation tree nodes?
                                            left=IsolationTree(height_limit=self.height_limit-1).fit(X_l, improved),
                                            right=IsolationTree(height_limit=self.height_limit-1).fit(X_r, improved),
                                            split_attr=q,
                                            split_value=p,
                                            n_rows=len(X))
                self.root.left.n_rows = len(X_l)
                self.root.right.n_rows = len(X_r)
                print(f"self.root.n_nodes = {self.root.n_nodes}")
                print(f"self.root.left.n_nodes = {self.root.left.n_nodes}")
                print(f"self.root.right.n_nodes = {self.root.right.n_nodes}")

                # print(f"self.root.n_rows = {self.root.n_rows}")
                # print(f"self.root.left.n_rows = {self.root.left.n_rows}")
                # print(f"self.root.right.n_rows = {self.root.right.n_rows}")
                self.n_nodes += self.root.left.n_nodes + self.root.right.n_nodes
        #print(f"self.root.n_nodes = {self.root.n_nodes}")    
        return self.root


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    ...
    # TODO: fill this out
    score_threshold = 1.0
    TPR = 0
    FPR = 0
    increment = 0.01
    print(f"desired_TPR = {desired_TPR}")
    while TPR < desired_TPR and (score_threshold - increment) > 0:
        score_threshold -= increment
        y_pred = np.array([1 if score < score_threshold else 0 for score in scores])
        confusion = confusion_matrix(y, y_pred)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        print(f"score_threshold = {score_threshold}, TPR = {TPR}, FPR = {FPR}")
        
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