import numpy as np
import random

class TreeNode:
    def __init__(self, feature: int = None, threshold: float = None, left: 'TreeNode' = None, 
                 right: 'TreeNode' = None, value: int = None, cost: float = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.cost = cost

class RegressionTree:
    def __init__(self, X, y, max_depth=10, min_samples_split=5, min_samples_leaf = 1, n_features=None):
        if len(X.shape) == 1:
            self.X = X = X.reshape(-1, 1)
        if n_features is None:
            n_features = np.shape(X)[1]
        if n_features > np.shape(X)[1]:
            raise ValueError(f"The maximum number of features is {np.shape(X)[1]}")
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.min_samples_leaf = min_samples_leaf
        self.feature_importance = np.zeros(self.X.shape[1])   
        self.n_features = n_features
    def _MSE(self, y):
        MSE = np.mean((y - np.mean(y))**2)
        return MSE
    def _VarianceReduction(self, X, y, threshold):
        # Calculates the variance reduction given a threshold.
        left_mask = X <= threshold
        right_mask = ~left_mask

        left_size = len(y[left_mask])
        right_size = len(y[right_mask])

        if left_size == 0 or right_size == 0:
            return -np.inf, -np.inf

        H_left = self._MSE(y[left_mask])
        H_right = self._MSE(y[right_mask])

        cost_left = (left_size / len(y)) * H_left 
        cost_right = (right_size / len(y)) * H_right
        
        return cost_left, cost_right 
    def _find_best_split(self, X, y, H):
        # Finds the best feature and threshold to split the data.
        
        possible_features_indexes = random.sample(range(0,np.shape(X)[1]), self.n_features)

        best_feature, best_threshold, best_variance_reduction = None, None, -np.inf
        best_cost_left, best_cost_right = None, None

        for feature in possible_features_indexes:
            tmp = np.sort(np.unique(X[:,feature]))
           #possible_thresholds = [np.mean(tmp[i:i+2]) for i in range(len(tmp)-1)]
            possible_thresholds = np.histogram_bin_edges(tmp, bins=20)[1:-1]
            
            for threshold in possible_thresholds:
                cost_left, cost_right = self._VarianceReduction(X[:,feature], y, threshold)
                variance_reduction = H - (cost_left + cost_right)
                if variance_reduction > best_variance_reduction:
                    best_feature, best_threshold, best_variance_reduction = feature, threshold, variance_reduction   
                    best_cost_left = cost_left
                    best_cost_right = cost_right
        return {"feature_number": best_feature, "threshold": best_threshold, "best_variance_reduction": best_variance_reduction,
                "cost_left": best_cost_left, "cost_right": best_cost_right}
    def _build_tree(self, X, y, cost, depth=0):
         # Recursively builds the decision tree.
        if len(np.unique(y)) == 1:
            return TreeNode(value=np.unique(y)[0], cost = cost)
        
        if depth >= self.max_depth or len(y) <= self.min_samples_split:
            return TreeNode(value=np.mean(y), cost = cost)
        

        best_feature, best_threshold, best_variance_reduction, cost_left, cost_right = self._find_best_split(X, y, cost).values()

        if best_variance_reduction == -np.inf or best_variance_reduction == np.inf or best_variance_reduction < 0:
            return TreeNode(value=np.mean(y), cost = cost) 
        
        
        self.feature_importance[best_feature] += best_variance_reduction / (depth + 1)
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        if len(y[left_mask]) < self.min_samples_leaf or len(y[right_mask]) < self.min_samples_leaf:
            return TreeNode(value=np.mean(y), cost = cost)

        left_node = self._build_tree(X = X[left_mask], y = y[left_mask], depth = depth + 1, cost = cost_left)
        right_node = self._build_tree(X = X[right_mask], y = y[right_mask], depth = depth + 1, cost = cost_right)

        return TreeNode(best_feature, best_threshold, left_node, right_node, cost = cost)
    def fit(self):
        # Trains the decision tree.
        self.root = self._build_tree(X = self.X, y = self.y, cost = self._MSE(self.y))
        total_importance = np.sum(self.feature_importance)
        if total_importance > 0:
            self.feature_importance /= total_importance
    def _predict(self, X):
         # Predicts a single sample.
        tmp = self.root
        while tmp.value is None:
            if X[tmp.feature] <= tmp.threshold:
                tmp = tmp.left
            else:
                tmp = tmp.right
        return tmp.value
    def predict(self, X):
        # Predicts a list of samples.
        if len(X.shape) == 1:
            return self._predict(X)
        else:
            return np.array([self._predict(i) for i in X])







