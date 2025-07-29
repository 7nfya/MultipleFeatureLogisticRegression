##### note: I used ChatGPT to organize the readme.
# Logistic Regression From Scratch (Multi-Feature Support)

This is a clean and educational implementation of **Logistic Regression from scratch** using only NumPy. It's designed to work with any dataset (binary classification) that contains numeric features.

You can use this code to **train on your own dataset** — just make sure to format your data correctly.

---

## What It Does

- Supports datasets with **multiple numerical features**
- Custom **train-test split** with random seed control
- Manual **feature standardization**
- Gradient descent training with **custom learning rate and epochs**
- Overflow-safe sigmoid for better numerical stability
- Custom accuracy calculation
- Self-contained and highly readable — perfect for learning or tinkering

---

## How to Use It On Your Own Dataset

1. Format your data as:

    ```python
    X = [[feature1, feature2, ..., featureN], [...], ...]  
    Y = [0, 1, 0, 0, 1, ...]   # 0 is false, 1 is true
    ```

2. Initialize and train the model:

    ```python
    from logistic_regression import LogisticRegression

    model = LogisticRegression()
    x_train, y_train, x_test, y_test = model.split(X, Y, train=0.8, seed=42)

    model.train(x_train, y_train, alpha=0.01, epochs=3000)
    print(model.predict(x_test))
    ```

3. Adjust the seed in `split()` to explore different train-test shuffles — **this directly affects the resulting accuracy**.

---

## Notes

- The dataset must be binary-labeled (`0` or `1`)
- The `split()` method internally uses a seed to randomize the dataset. A better seed = better data separation = potentially higher accuracy.
- For best performance, try multiple seeds and monitor how the accuracy changes.

---

## Example Output
```python
Number of samples in the dataset: 569
Number of features in each entry: 30
First sample features: [17.99....]
First sample label: 0
training samples' length: 455
first Training samples: [12.1....]
testing samples' length: 114
first testing sample: [14.54....]
Accuracy: 99.12%
```
