from collections import Counter

import numpy as np


def _calc_entropy(x):
    counts = Counter(x.reshape(-1)).values()
    p_x = np.array([1. * cnt / len(x) for cnt in counts])
    return -(p_x * np.log(p_x)).sum()


def _entropy_info_gain(y, left_y, right_y):
    left_f = len(left_y) / len(y)
    right_f = len(right_y) / len(y)
    before = _calc_entropy(y)
    after = left_f * _calc_entropy(left_y) + right_f * _calc_entropy(right_y)
    info_gain = before - after
    return info_gain


def _calc_gini(x):
    counts = Counter(x.reshape(-1)).values()
    probs = np.array([1. * cnt / len(x) for cnt in counts])
    return 1.0 - np.sum(probs ** 2)


def _gini_info_gain(y, left_y, right_y):
    left_f = len(left_y) / len(y)
    right_f = len(right_y) / len(y)
    before = _calc_gini(y)
    after = left_f * _calc_gini(left_y) + right_f * _calc_gini(right_y)
    return before - after


def mse(a, b):
    return np.power(a - b, 2).mean()


def _mse_info_gain(y, left_y, right_y):
    left_f = len(left_y) / len(y)
    right_f = len(right_y) / len(y)
    before = mse(y, y.mean(0))
    after = left_f * mse(left_y, left_y.mean(axis=0)) + \
        right_f * mse(right_y, right_y.mean(axis=0))
    return np.mean(before - after)


class Node:

    """
    Args:
        feat_idx (int or None): An index of the feature the Node splits.
        threshold (float or None): A value point to split data.
        value (float or None): A target value of the Node(Tree).
        left, right (Node of None): A left, right child Node.
    """

    def __init__(
        self,
        feat_idx=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):
        self.feat_idx = feat_idx
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right


class BaseDecisionTree:

    """
    Args:
        criterion (str): 'entropy' or 'gini'.
        max_features (float): A number of features used to split node.
                              Default is 1.0.
        max_depth (int or None): A depth of built nodes. Default is 1e3.
        min_samples_split (int): A minimum sample size for each node.
        min_impurity_split (float): A threshold to stop building node.
    """
    def __init__(
        self,
        criterion: str = 'gini',
        max_features: float = 1.0,
        max_depth: int = 1e3,
        min_impurity_split: float = 1e-7,
        min_samples_split: int = 2
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_impurity_split = min_impurity_split
        self.min_samples_split = min_samples_split

        self.root = None
        self.feature_importances_ = None
        self.feature_scores_ = None

    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.root = self._build_tree(X, y)
        self.feature_importances_ = (
            self.feature_scores_ / self.feature_scores_.sum()
        )
        return self

    def _predict_sample(self, x, node=None):
        """ Make prediction on a sample.
        Args:
            x (np.ndarray): A sample, 1d array, to predict.
            node (Node or None): A current node. If node is None,
                                 root node is used.
        """
        if node is None:
            node = self.root

        # Prediction result.
        if node.value is not None:
            return node.value

        feat_value = x[node.feat_idx]
        node = node.left if feat_value < node.threshold else node.right
        return self._predict_sample(x, node=node)

    def predict(self, X):
        return np.array([self._predict_sample(x) for x in X])

    def _calc_info_gain(self):
        raise NotImplementedError

    def _aggregate_target(self):
        raise NotImplementedError

    def _build_tree(self, X, y, curr_depth=0):
        num_samples, num_features = X.shape
        self.feature_scores_ = np.zeros(num_features)

        best_score = 0
        has_enough_samples = num_samples >= self.min_samples_split
        less_than_max_depth = curr_depth <= self.max_depth
        if has_enough_samples and less_than_max_depth:
            split, best_score = self._search_best_split(X, y)

        if best_score > self.min_impurity_split:
            feat_idx = split['feat_idx']
            threshold = split['threshold']
            left_x, left_y = split['left_x'], split['left_y']
            right_x, right_y = split['right_x'], split['right_y']

            left = self._build_tree(left_x, left_y, curr_depth + 1)
            right = self._build_tree(right_x, right_y, curr_depth + 1)

            self.feature_scores_[feat_idx] += best_score

            return Node(feat_idx=feat_idx, threshold=threshold,
                        left=left, right=right)
        else:
            # In classification, use the most common value as target.
            # In Regression, just calculate mean of target.
            y_aggrigated = self._aggregate_target(y)
            return Node(value=y_aggrigated)

    def _search_best_split(self, X, y):
        xy = np.concatenate((X, y), axis=1)

        max_score = 0.0
        best_split = None

        num_features = X.shape[1]
        num_features_used_to_split = int(self.max_features * X.shape[1])
        feat_indices = np.random.choice(
            range(num_features), num_features_used_to_split
        )
        # Iterate over features.
        for feat_idx in feat_indices:
            # Iterate over unique feature values.
            for thr in np.unique(X[:, feat_idx]):
                left_indices = xy[:, feat_idx] < thr
                right_indices = xy[:, feat_idx] >= thr
                left_xy = xy[left_indices]
                right_xy = xy[right_indices]
                if len(left_xy) == 0 or len(right_xy) == 0:
                    continue
                left_y = left_xy[:, num_features:]
                right_y = right_xy[:, num_features:]
                score = self._calc_info_gain(y, left_y, right_y)
                if score > max_score:
                    max_score = score
                    best_split = {
                        'feat_idx': feat_idx,
                        'threshold': thr,
                        'left_x': left_xy[:, :num_features],
                        'right_x': right_xy[:, :num_features],
                        'left_y': left_y, 'right_y': right_y
                    }
        return best_split, max_score


class DecisionTreeClassifier(BaseDecisionTree):

    def __init__(
        self,
        criterion: str = 'gini',
        max_features: float = 1.0,
        max_depth: int = 1e3,
        min_impurity_split: float = 1e-7,
        min_samples_split: int = 2
    ):
        super().__init__(
            criterion, max_features, max_depth,
            min_impurity_split, min_samples_split
        )

    def _calc_info_gain(self, y, left_y, right_y):
        if self.criterion == 'entropy':
            return _entropy_info_gain(y, left_y, right_y)
        else:
            return _gini_info_gain(y, left_y, right_y)

    def _aggregate_target(self, y):
        res = Counter(y.reshape(-1))
        return res.most_common()[0][0]


class DecisionTreeRegressor(BaseDecisionTree):

    def __init__(
        self,
        criterion: str = 'mse',
        max_features: float = 1.0,
        max_depth: int = 1e3,
        min_impurity_split: float = 1e-7,
        min_samples_split: int = 2
    ):
        super().__init__(
            criterion, max_features, max_depth,
            min_impurity_split, min_samples_split
        )

    def _calc_info_gain(self, y, left_y, right_y):
        if self.criterion == 'mse':
            return _mse_info_gain(y, left_y, right_y)
        else:
            raise ValueError('criterion must be mse.')

    def _aggregate_target(self, y):
        return np.mean(y)
