# Regression Tree Model

This project implements a Regression Tree algorithm in Python, focusing on regression tasks. The tree is built using recursive binary splits based on variance reduction.

## Features:
- Recursive construction of a regression tree
- Variance reduction-based node splits
- Feature importance calculation based on variance reduction
- Hyperparameter tuning (`max_depth`, `min_samples_split`, `min_samples_leaf`)

## How it works:
- The algorithm calculates the Mean Squared Error (MSE) of the target variable `y`.
- It splits the dataset based on different thresholds for each feature.
- The split that results in the maximum reduction in variance (variance reduction) is chosen.
- This process is recursively applied to subtrees, limited by parameters like `max_depth`, `min_samples_split`, and `min_samples_leaf`.
