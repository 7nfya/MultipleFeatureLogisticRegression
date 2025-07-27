# Multiple Feature Logistic Regression
## This is a lightweight logistic regression model built from scratch using only np no scikit-learn or external libraries. It's designed to handle binary classification tasks using any number of features, and  it uses gradient descent for training.

This project can serve as a learning tool for understanding how logistic regression works under the hood.

Features
- Custom implementation of logistic regression

-  Works with multiple input features

- Fully implemented gradient descent

- Binary classification (0 or 1)

- Includes prediction after training

- Easy to extend or modify

How It Works
- Input your feature matrix X and corresponding binary labels y

- Initialize the model

- Train using .train() method

- Predict using .predict() on new datafunction:

```python
X = [
    [1, 2],
    [2, 3],
    [3, 1],
    [6, 5],
    [7, 8],
    [8, 6]
]
y = [0, 0, 0, 1, 1, 1]

model = MultipleFeatureLogisticRegression(X, y)
model.train()
print(model.predict([4, 4]))  # Output: 0 or 1
```
Behind the Scenes
- The model uses the sigmoid function to convert raw outputs into probabilities.

- A custom training loop updates weights and bias using batch gradient descent.

- A safeguard (TrainTheModelError) ensures prediction can’t happen before training.

Visualization:
- If you're interested in visualizing decision boundaries for 2D data, you can easily integrate this with matplotlib. This implementation is kept clean and minimal on purpose.


Author
Omar Nesr Edin

GitHub: 7nfya

Instagram: @omar.vae

